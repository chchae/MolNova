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


def prepare_single_elite_scaffold(elite, previous, docking_smarts):
    """Extend each preceding attachment's sidechain at one C-H of an elite."""
    if previous is None:
        raise RuntimeError('Single-elite combichem requires the preceding iteration scaffold')
    mol = Chem.RemoveHs(Chem.Mol(elite['mol']))
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(smiles)
    old_core = Chem.MolFromSmarts(previous['mcs_smarts'])
    old_scaffold = Chem.MolFromSmiles(previous['scaffold_smiles'])
    docking = Chem.MolFromSmarts(docking_smarts)
    if (old_core is None or not old_core.GetNumHeavyAtoms() or old_scaffold is None
            or docking is None or not docking.GetNumHeavyAtoms()):
        raise RuntimeError('Invalid preceding scaffold/core or docking core for single-elite combichem')
    old_attachments = attachment_atoms(old_scaffold)
    if sorted(old_scaffold.GetAtomWithIdx(i).GetAtomMapNum() for i in old_attachments) != list(range(len(old_attachments))):
        raise RuntimeError('Preceding combichem scaffold needs consecutive attachment labels starting at 0')
    if (not old_scaffold.HasSubstructMatch(old_core, useChirality=False)
            or not old_scaffold.HasSubstructMatch(docking, useChirality=False)
            or not mol.HasSubstructMatch(docking, useChirality=False)
            or _core.decompose_elite_for_libinvent(smiles, previous['scaffold_smiles']) is None):
        raise RuntimeError('Single elite is incompatible with the preceding scaffold or fixed docking core')
    query, by_label, _, dummy_indices, _ = _core._scaffold_attachment_layout(previous['scaffold_smiles'])
    match = mol.GetSubstructMatch(query, useChirality=False)
    core_atoms = {index for qidx, index in enumerate(match) if qidx not in dummy_indices}
    candidate_groups = []
    for label in range(len(old_attachments)):
        sidechain = set()
        pending = [match[by_label[label]['dummy_qidx']]]
        while pending:
            index = pending.pop()
            if index in core_atoms or index in sidechain:
                continue
            sidechain.add(index)
            pending.extend(atom.GetIdx() for atom in mol.GetAtomWithIdx(index).GetNeighbors())
        candidates = [index for index in sorted(sidechain)
                      if mol.GetAtomWithIdx(index).GetAtomicNum() == 6
                      and mol.GetAtomWithIdx(index).GetTotalNumHs() > 0]
        if not candidates:
            raise RuntimeError(f'Single-elite combichem found no sidechain C-H expansion site '
                               f'for preceding attachment {label}')
        candidate_groups.append(candidates)
    sites = [secrets.choice(candidates) for candidates in candidate_groups]
    # Replace a real hydrogen so RDKit retains the ordering at stereocenters.
    expanded = Chem.AddHs(mol)
    for label, site in enumerate(sites):
        hydrogen = next(atom for atom in expanded.GetAtomWithIdx(site).GetNeighbors()
                        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 0)
        hydrogen.SetAtomicNum(0)
        hydrogen.SetAtomMapNum(label)
    scaffold = Chem.RemoveHs(expanded)
    Chem.SanitizeMol(scaffold)
    attachment_atoms(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=True)
    import re
    scaffold_smiles = re.sub(r'(?<!\[)\*', '[*:0]', scaffold_smiles)
    # The generation core describes the retained heavy-atom skeleton. Hydrogen
    # counts at the expansion carbon change when its hydrogen becomes a bond.
    core = Chem.Mol(mol)
    for site in sites:
        core.GetAtomWithIdx(site).SetNumExplicitHs(0)
    smarts = Chem.MolToSmarts(core)
    if not scaffold.HasSubstructMatch(Chem.MolFromSmarts(smarts), useChirality=False):
        raise RuntimeError('Single-elite expansion does not retain its generation core')
    return dict(mcs_smarts=smarts, scaffold_smiles=scaffold_smiles,
                source='single_elite_sidechain_expansion', source_ids=[elite['id']],
                expansion_atom_indices=sites, expansion_elite_smiles=smiles,
                expansion_candidate_indices=candidate_groups)


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
