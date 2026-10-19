"""Local sidechain assembly and reproducible, bounded Cartesian enumeration."""
import csv
import hashlib
import heapq
import math
from pathlib import Path
import random
import secrets

from rdkit import Chem, rdBase

from molnova import _core, database


def attachment_atoms(mol):
    """Return terminal single-bond dummy indices in stable atom order."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('Expected one connected molecule')
    dummies = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if not dummies or any(atom.GetDegree() != 1 or
                          atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE or
                          atom.GetNeighbors()[0].GetAtomicNum() <= 1 for atom in dummies):
        raise ValueError('Expected terminal dummy atoms attached to heavy atoms by single bonds')
    return [atom.GetIdx() for atom in dummies]


def prepare_scaffold(inputs, supplied, *, mcs_timeout=60):
    """Accept supplied attachments, or add every inferred boundary to the core."""
    smarts = supplied if supplied is not None else _core.infer_mcs_smarts_from_iteration0(
        inputs, timeout=mcs_timeout)
    query = Chem.MolFromSmarts(smarts)
    if query is None or not query.GetNumHeavyAtoms():
        raise ValueError('Invalid combichem generation core')
    matched = [ref for ref in inputs if ref['mol'].HasSubstructMatch(query, useChirality=False)]
    if not matched:
        raise RuntimeError('Combichem generation core matches none of the input ligands')
    if any('AtomNull' in atom.DescribeQuery() for atom in query.GetAtoms()):
        # Explicit scaffolds must describe a concrete molecule, as for LibInvent.
        scaffold = Chem.MolFromSmiles(smarts)
        dummies = attachment_atoms(scaffold)
        for atom in scaffold.GetAtoms():
            atom.SetAtomMapNum(0)
        for label, index in enumerate(dummies):
            scaffold.GetAtomWithIdx(index).SetAtomMapNum(label)
        scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=True)
        import re
        scaffold_smiles = re.sub(r'(?<!\[)\*', '[*:0]', scaffold_smiles)
    else:
        sites = _core.determine_libinvent_attachment_sites(matched, smarts, all_sites=True)
        # Reject non-single boundaries rather than changing a bond's chemistry.
        for ref in matched:
            mol, match = _core.get_first_core_match(ref['mol'], query)
            core = set(match)
            if any(bond.GetBondType() != Chem.BondType.SINGLE
                   for index in match for bond in mol.GetAtomWithIdx(index).GetBonds()
                   if (bond.GetBeginAtomIdx() in core) != (bond.GetEndAtomIdx() in core)):
                raise RuntimeError('Combichem inferred attachment boundary is not a single bond')
        scaffold_smiles = _core.build_libinvent_scaffold(matched, smarts, sites, require_two=False)
    attachment_atoms(Chem.MolFromSmiles(scaffold_smiles))
    return {'mcs_smarts': smarts, 'scaffold_smiles': scaffold_smiles}


def load_library(path, report):
    """Validate, canonicalize and deduplicate single-ended fragments."""
    path, report = Path(path), Path(report)
    digest = hashlib.sha256()
    accepted, seen = [], set()
    rejected = 0
    with path.open('rb') as stream, report.open('w', newline='') as output:
        writer = csv.writer(output, delimiter='\t')
        writer.writerow(['line', 'smiles', 'reason'])
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            line = raw.decode('utf-8').strip()
            if not line or line.startswith('#'):
                continue
            smiles = line.split()[0]
            try:
                with rdBase.BlockLogs():
                    mol = Chem.MolFromSmiles(smiles)
                dummies = attachment_atoms(mol)
                if len(dummies) != 1:
                    raise ValueError('sidechain must have exactly one attachment')
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                if canonical in seen:
                    raise ValueError('duplicate sidechain')
                seen.add(canonical)
                accepted.append({'line': line_number, 'smiles': canonical,
                                 'heavy_atoms': mol.GetNumHeavyAtoms()})
            except ValueError as error:
                rejected += 1
                writer.writerow([line_number, smiles, str(error)])
    if not accepted:
        raise RuntimeError(f'No eligible sidechains in {path}; see {report}')
    return accepted, digest.hexdigest(), rejected


def select_sidechains(library, rng, count=1000):
    """Weighted sampling without replacement, weight = 1 / heavy-atom count."""
    # Exponential races avoid probability underflow and require no extra package.
    chosen = heapq.nsmallest(min(count, len(library)),
                            ((-math.log(1.0 - rng.random()) * row['heavy_atoms'], index)
                             for index, row in enumerate(library)))
    selected = [dict(library[index]) for _, index in chosen]
    rng.shuffle(selected)
    return selected


def prepare_plan(args, iteration, directory):
    saved = database.combichem_plan(args.db_path, iteration)
    scaffold = args.libinvent_scaffold_smiles
    if saved is not None:
        if saved['scaffold_smiles'] != scaffold:
            raise RuntimeError('Saved combichem scaffold differs from the iteration snapshot')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    library, fingerprint, rejected = load_library(
        args.sidechain_library, directory / 'sidechain_rejected.tsv')
    seed = secrets.randbits(128)
    rng = random.Random(seed)
    molecule = Chem.MolFromSmiles(scaffold)
    dummies = attachment_atoms(molecule)
    pools = [select_sidechains(library, rng) for _ in dummies]
    total = math.prod(len(pool) for pool in pools)
    # A random coprime stride visits every Cartesian index exactly once without
    # allocating a shuffled list that could contain billions of combinations.
    stride = rng.randrange(1, total) if total > 1 else 1
    while math.gcd(stride, total) != 1:
        stride = rng.randrange(1, total)
    plan = {'protocol': 'combichem_v1', 'scaffold_smiles': scaffold,
            'library_path': str(args.sidechain_library), 'library_sha256': fingerprint,
            'eligible_sidechains': len(library), 'rejected_sidechains': rejected,
            'seed': str(seed), 'selection_weight': '1/heavy_atoms', 'pools': pools,
            'attachment_labels': [molecule.GetAtomWithIdx(index).GetAtomMapNum() for index in dummies],
            'total_combinations': str(total), 'offset': str(rng.randrange(total)),
            'stride': str(stride), 'batch_size': args.sample_size}
    return database.combichem_plan(args.db_path, iteration, plan)


def combination_at(plan, position):
    """Decode a saved permutation of a mixed-radix Cartesian product."""
    total = int(plan['total_combinations'])
    if not 0 <= position < total:
        raise IndexError('Combination position outside saved plan')
    index = (int(plan['offset']) + position * int(plan['stride'])) % total
    indices = []
    for pool in reversed(plan['pools']):
        index, digit = divmod(index, len(pool))
        indices.append(digit)
    return [pool[digit] for pool, digit in zip(plan['pools'], reversed(indices))]


def assemble(scaffold, sidechains):
    """Zip matched dummy atoms, retaining existing atom/bond stereochemistry."""
    core = Chem.Mol(scaffold)
    dummies = attachment_atoms(core)
    if len(dummies) != len(sidechains):
        raise ValueError('Sidechain count differs from scaffold attachment count')
    for atom in core.GetAtoms():
        atom.SetAtomMapNum(0)
    for label, index in enumerate(dummies, 1):
        core.GetAtomWithIdx(index).SetAtomMapNum(label)
    for label, sidechain in enumerate(sidechains, 1):
        fragment = Chem.Mol(sidechain)
        fragment_dummies = attachment_atoms(fragment)
        if len(fragment_dummies) != 1:
            raise ValueError('Expected a single-ended sidechain')
        for atom in fragment.GetAtoms():
            atom.SetAtomMapNum(0)
        fragment.GetAtomWithIdx(fragment_dummies[0]).SetAtomMapNum(label)
        core = Chem.CombineMols(core, fragment)
    try:
        with rdBase.BlockLogs():
            product = Chem.molzip(core)
            Chem.SanitizeMol(product)
        if any(atom.GetAtomicNum() == 0 for atom in product.GetAtoms()):
            return None
        for atom in product.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
    except (RuntimeError, ValueError):
        return None


def write_batch(plan, start, stop, directory):
    """Materialize only one batch, with sidechain provenance for every attempt."""
    scaffold = Chem.MolFromSmiles(plan['scaffold_smiles'])
    # Parse each selected fragment once per batch, rather than once per product.
    molecules = {row['smiles']: Chem.MolFromSmiles(row['smiles'])
                 for pool in plan['pools'] for row in pool}
    output = directory / f'generated-{start}-{stop}.csv'
    audit = directory / f'assembly-{start}-{stop}.tsv'
    with output.open('w', newline='') as stream, audit.open('w', newline='') as log:
        writer, reporter = csv.writer(stream), csv.writer(log, delimiter='\t')
        writer.writerow(['SMILES'])
        reporter.writerow(['combination', 'sidechain_lines', 'smiles', 'status'])
        for position in range(start, stop):
            rows = combination_at(plan, position)
            smiles = assemble(scaffold, [molecules[row['smiles']] for row in rows])
            reporter.writerow([position, ','.join(str(row['line']) for row in rows),
                               smiles or '', 'assembled' if smiles else 'invalid_product'])
            if smiles:
                writer.writerow([smiles])
    return output
