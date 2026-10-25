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


class NonSingleAttachmentBoundaryError(RuntimeError):
    """An inferred core would require replacing a non-single bond."""

    def __init__(self, mcs_smarts):
        super().__init__('Combichem inferred attachment boundary is not a single bond')
        self.mcs_smarts = mcs_smarts


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
                raise NonSingleAttachmentBoundaryError(smarts)
        scaffold_smiles = _core.build_libinvent_scaffold(matched, smarts, sites, require_two=False)
    attachment_atoms(Chem.MolFromSmiles(scaffold_smiles))
    return {'mcs_smarts': smarts, 'scaffold_smiles': scaffold_smiles}


def _terminal_cuts(mol, sidechain, core_atoms):
    """Single-bond cuts whose distal component belongs only to this sidechain."""
    candidates = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing()
                or not ({a, b} & sidechain)
                or mol.GetAtomWithIdx(a).GetAtomicNum() <= 1
                or mol.GetAtomWithIdx(b).GetAtomicNum() <= 1):
            continue
        component = set()
        pending = [a]
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend(atom.GetIdx() for atom in mol.GetAtomWithIdx(index).GetNeighbors()
                           if {index, atom.GetIdx()} != {a, b})
        if component & core_atoms:
            retained, distal = a, b
            removed = set(range(mol.GetNumAtoms())) - component
        else:
            retained, distal = b, a
            removed = component
        if removed and removed <= sidechain:
            candidates.append(dict(bond_index=bond.GetIdx(), retained_atom_index=retained,
                                   distal_atom_index=distal, removed_atom_indices=sorted(removed)))
    return candidates


def _retained_core_query(scaffold):
    # SMARTS serialization may reorder atoms: remove dummies using parsed indices.
    core = Chem.RWMol(Chem.MolFromSmarts(Chem.MolToSmarts(scaffold)))
    for index in sorted((atom.GetIdx() for atom in core.GetAtoms() if atom.GetAtomicNum() == 0), reverse=True):
        core.RemoveAtom(index)
    return core.GetMol()


def prepare_single_elite_scaffold(elite, previous, docking_smarts, *, selected_cuts=None):
    """Replace a terminal part of each preceding attachment's elite sidechain."""
    if previous is None:
        raise RuntimeError('Single-elite combichem requires the preceding iteration scaffold')
    mol = Chem.RemoveHs(Chem.Mol(elite['mol']))
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    mol = Chem.MolFromSmiles(smiles)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise RuntimeError('Single-elite combichem requires a connected elite')
    old_core = Chem.MolFromSmarts(previous['mcs_smarts'])
    old_scaffold = Chem.MolFromSmiles(previous['scaffold_smiles'])
    if (previous.get('source') == 'single_elite_sidechain_replacement' and old_core is not None
            and old_scaffold is not None and any(atom.GetAtomicNum() == 0 for atom in old_core.GetAtoms())):
        # Repair the interpretation of legacy dummy-index serialization without
        # rewriting its immutable snapshot or changing the recorded scaffold.
        old_core = _retained_core_query(old_scaffold)
    docking = Chem.MolFromSmarts(docking_smarts)
    if (old_core is None or not old_core.GetNumHeavyAtoms() or old_scaffold is None
            or docking is None or not docking.GetNumHeavyAtoms()):
        raise RuntimeError('Invalid preceding scaffold/core or docking core for single-elite combichem')
    old_attachments = attachment_atoms(old_scaffold)
    if sorted(old_scaffold.GetAtomWithIdx(i).GetAtomMapNum() for i in old_attachments) != list(range(len(old_attachments))):
        raise RuntimeError('Preceding combichem scaffold needs consecutive attachment labels starting at 0')
    decomposition = _core._match_elite_to_scaffold(mol, previous['scaffold_smiles'])
    if (not old_scaffold.HasSubstructMatch(old_core, useChirality=False)
            or not old_scaffold.HasSubstructMatch(docking, useChirality=False)
            or not mol.HasSubstructMatch(docking, useChirality=False)
            or decomposition is None):
        raise RuntimeError('Single elite is incompatible with the preceding scaffold or fixed docking core')
    _, by_label, _, dummy_indices, _ = _core._scaffold_attachment_layout(previous['scaffold_smiles'])
    match = decomposition[1]
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
        candidates = _terminal_cuts(mol, sidechain, core_atoms)
        if not candidates:
            raise RuntimeError(f'Single-elite combichem found no terminal single-bond replacement '
                               f'for preceding attachment {label}')
        candidate_groups.append(candidates)
    if selected_cuts is not None:
        if (len(selected_cuts) != len(candidate_groups)
                or any(cut not in candidates for cut, candidates in zip(selected_cuts, candidate_groups))):
            raise RuntimeError('Saved attachment cuts are incompatible with the elite or preceding scaffold')
        cuts = selected_cuts
    else:
        cuts = [secrets.choice(candidates) for candidates in candidate_groups]
    # RDKit inserts a dummy at each cut endpoint, preserving retained stereo.
    fragmented = Chem.FragmentOnBonds(mol, [cut['bond_index'] for cut in cuts],
                                     dummyLabels=[(label, label) for label in range(len(cuts))])
    mappings = []
    fragments = Chem.GetMolFrags(fragmented, asMols=True, fragsMolAtomMapping=mappings)
    scaffold = next(fragment for fragment, indices in zip(fragments, mappings)
                    if core_atoms <= set(indices))
    for atom in scaffold.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(atom.GetIsotope())
            atom.SetIsotope(0)
    attachments = attachment_atoms(scaffold)
    if sorted(scaffold.GetAtomWithIdx(i).GetAtomMapNum() for i in attachments) != list(range(len(cuts))):
        raise RuntimeError('Single-elite replacement did not preserve the preceding attachment labels')
    if (not scaffold.HasSubstructMatch(old_core, useChirality=False)
            or not scaffold.HasSubstructMatch(docking, useChirality=False)):
        raise RuntimeError('Single-elite replacement does not contain the preceding and fixed docking cores')
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=True)
    import re
    scaffold_smiles = re.sub(r'(?<!\[)\*', '[*:0]', scaffold_smiles)
    # Remove query dummies to describe only the retained heavy-atom core,
    # without inventing hydrogen counts by capping the cut positions.
    core = _retained_core_query(scaffold)
    smarts = Chem.MolToSmarts(core)
    if not scaffold.HasSubstructMatch(Chem.MolFromSmarts(smarts), useChirality=False):
        raise RuntimeError('Single-elite replacement does not retain its generation core')
    return dict(mcs_smarts=smarts, scaffold_smiles=scaffold_smiles,
                source='single_elite_sidechain_replacement', source_ids=[elite['id']],
                replacement_cuts=cuts, replacement_elite_smiles=smiles,
                replacement_candidates=candidate_groups)


def prepare_ranked_elite_scaffold(ranked_inputs, previous, docking_smarts, *,
                                  mcs_timeout=60, single_previous=None):
    """Halve a score-ranked prefix until its MCS safely extends the saved core."""
    if not ranked_inputs or previous is None:
        raise RuntimeError('Adaptive combichem requires ranked elites and the preceding scaffold')
    old_scaffold = Chem.MolFromSmiles(previous['scaffold_smiles'])
    attachment_atoms(old_scaffold)
    old_core = _retained_core_query(old_scaffold)
    docking = Chem.MolFromSmarts(docking_smarts)
    if docking is None or not docking.GetNumHeavyAtoms() or not old_scaffold.HasSubstructMatch(docking):
        raise RuntimeError('Preceding combichem scaffold does not retain the fixed docking core')
    attempts = []
    count = len(ranked_inputs)
    while count > 1:
        selected = ranked_inputs[:count]
        attempt = dict(elite_count=count, source_ids=[ref['id'] for ref in selected])
        print(f'[MCS] Trying highest-ranked {count}/{len(ranked_inputs)} elites '
              f'(timeout={mcs_timeout}s).', flush=True)
        try:
            info = prepare_scaffold(selected, None, mcs_timeout=mcs_timeout)
        except _core.MCSSearchTimeoutError as error:
            # Reject the canceled attempt without accepting or recording its
            # partial SMARTS. Other MCS errors must still stop generation.
            attempt.update(status='mcs_timeout', mcs_smarts=None,
                           timeout_seconds=mcs_timeout, failure_message=str(error))
        except (NonSingleAttachmentBoundaryError, _core.InsufficientAttachmentSitesError) as error:
            attempt.update(status=('non_single_attachment_boundary' if isinstance(
                error, NonSingleAttachmentBoundaryError) else 'insufficient_attachment_sites'),
                mcs_smarts=error.mcs_smarts)
        else:
            scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
            # Compare chemical graphs, independent of dummy labels, SMARTS atom
            # ordering and RDKit ring-query spelling. No bonds are converted.
            core = _retained_core_query(scaffold)
            attempt.update(mcs_smarts=info['mcs_smarts'], core_heavy_atoms=core.GetNumHeavyAtoms())
            if core.GetNumHeavyAtoms() <= old_core.GetNumHeavyAtoms():
                attempt['status'] = 'core_not_larger'
            elif not core.HasSubstructMatch(old_core, useChirality=False):
                attempt['status'] = 'previous_core_not_retained'
            elif not scaffold.HasSubstructMatch(docking, useChirality=False):
                attempt['status'] = 'docking_core_not_retained'
            elif any(_core.decompose_elite_for_libinvent(
                    Chem.MolToSmiles(ref['mol'], isomericSmiles=True), info['scaffold_smiles']) is None
                    for ref in selected):
                attempt['status'] = 'elite_decomposition_failed'
            else:
                attempt['status'] = 'expanded_core'
                print(f'[MCS] {count} elites: expanded_core.', flush=True)
                attempts.append(attempt)
                result = dict(info, source='previous_iteration_elites' if count == len(ranked_inputs)
                              else 'ranked_elite_subset_mcs', source_ids=attempt['source_ids'])
                break
        attempts.append(attempt)
        print(f"[MCS] {count} elites: {attempt['status']}; "
              f'trying highest-ranked {max(1, count // 2)}.', flush=True)
        count = max(1, count // 2)
    else:
        # A one-molecule MCS has no variable boundary. Preserve the existing
        # terminal-sidechain replacement protocol and its durable cut choices.
        result = prepare_single_elite_scaffold(ranked_inputs[0], single_previous or previous, docking_smarts)
        attempts.append(dict(elite_count=1, source_ids=[ranked_inputs[0]['id']],
                             status='single_elite_replacement'))
    result.update(elite_mcs_policy='ranked_halving_v2', elite_mcs_attempts=attempts,
                  ranked_elite_ids=[ref['id'] for ref in ranked_inputs],
                  previous_core_heavy_atoms=old_core.GetNumHeavyAtoms())
    return result


def load_library(path, report, max_heavy_atoms=None):
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
                heavy_atoms = mol.GetNumHeavyAtoms()
                if max_heavy_atoms is not None and heavy_atoms > max_heavy_atoms:
                    raise ValueError(f'sidechain heavy atoms {heavy_atoms} exceeds maximum {max_heavy_atoms}')
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
                if canonical in seen:
                    raise ValueError('duplicate sidechain')
                seen.add(canonical)
                accepted.append({'line': line_number, 'smiles': canonical,
                                 'heavy_atoms': heavy_atoms})
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


def _draw_plan(scaffold, library, fingerprint, rejected, library_path, batch_size, max_heavy_atoms=None):
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
    return {'protocol': 'combichem_v1', 'scaffold_smiles': scaffold,
            'library_path': str(library_path), 'library_sha256': fingerprint,
            'sidechain_max_heavy_atoms': max_heavy_atoms,
            'eligible_sidechains': len(library), 'rejected_sidechains': rejected,
            'seed': str(seed), 'selection_weight': '1/heavy_atoms', 'pools': pools,
            'attachment_labels': [molecule.GetAtomWithIdx(index).GetAtomMapNum() for index in dummies],
            'total_combinations': str(total), 'offset': str(rng.randrange(total)),
            'stride': str(stride), 'batch_size': batch_size}


def prepare_plan(args, iteration, directory):
    saved = database.combichem_plan(args.db_path, iteration)
    scaffold = args.libinvent_scaffold_smiles
    if saved is not None:
        if saved.get('base_scaffold_smiles', saved['scaffold_smiles']) != scaffold:
            raise RuntimeError('Saved combichem scaffold differs from the iteration snapshot')
        if saved.get('attachment_selection', {}).get('scaffold_smiles', saved['scaffold_smiles']) != saved['scaffold_smiles']:
            raise RuntimeError('Saved combichem round scaffold differs from its attachment selection')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    maximum = getattr(args, 'sidechain_max_heavy_atoms', None)
    library, fingerprint, rejected = load_library(
        args.sidechain_library, directory / 'sidechain_rejected.tsv', maximum)
    plan = _draw_plan(scaffold, library, fingerprint, rejected, args.sidechain_library, args.sample_size, maximum)
    plan['round'] = 0
    return database.combichem_plan(args.db_path, iteration, plan)


def _next_attachment_selection(args, iteration):
    """Choose uniformly from untried cut combinations without expanding their product."""
    original = database.iteration_scaffold(args.db_path, iteration)
    if original is None or original.get('source') != 'single_elite_sidechain_replacement':
        raise RuntimeError('Attachment switching requires an adaptive single-elite replacement scaffold')
    groups = original['replacement_candidates']
    positions = [{cut['bond_index']: i for i, cut in enumerate(group)} for group in groups]
    tried = [original]
    tried.extend(plan.get('attachment_selection', original)
                 for plan in database.combichem_plans(args.db_path, iteration))
    used = set()
    for selection in tried:
        rank = 0
        for group, mapping, cut in zip(groups, positions, selection['replacement_cuts']):
            rank = rank * len(group) + mapping[cut['bond_index']]
        used.add(rank)
    total = math.prod(len(group) for group in groups)
    if len(used) == total:
        raise RuntimeError('All single-elite attachment combinations have been tried; generation remains incomplete')
    rank = secrets.randbelow(total - len(used))
    for used_rank in sorted(used):
        if used_rank > rank:
            break
        rank += 1
    cuts = []
    for group in reversed(groups):
        rank, digit = divmod(rank, len(group))
        cuts.append(group[digit])
    cuts.reverse()
    previous = (database.combichem_compound_scaffold(args.db_path, original['source_ids'][0])
                or database.iteration_scaffold(args.db_path, iteration - 1))
    selection = prepare_single_elite_scaffold(
        dict(id=original['source_ids'][0], mol=Chem.MolFromSmiles(original['replacement_elite_smiles'])),
        previous, original['docking_mcs_smarts'], selected_cuts=cuts,
    )
    selection.update(replacement_from_iteration=iteration - 1, docking_mcs_smarts=original['docking_mcs_smarts'])
    return original, selection


def refill_plan(args, iteration, directory, previous, *, change_attachments=False):
    """Add a saved round without changing earlier pools or scientific results."""
    selection = None
    if change_attachments:
        original, selection = _next_attachment_selection(args, iteration)
    if not change_attachments and all(len(pool) == previous['eligible_sidechains'] for pool in previous['pools']):
        count = database.iteration_compound_count(args.db_path, iteration)
        raise RuntimeError(f'Iteration {iteration}: combichem combinations exhausted at '
                           f'{count}/{args.target_count} compounds; all eligible sidechains '
                           'already covered; generation remains incomplete.')
    round_id = previous.get('round', 0) + 1
    folder = directory / f'round-{round_id:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    library_path = Path(previous['library_path'])
    maximum = previous.get('sidechain_max_heavy_atoms')
    library, fingerprint, rejected = load_library(library_path, folder / 'sidechain_rejected.tsv', maximum)
    if fingerprint != previous['library_sha256']:
        raise RuntimeError('Combichem refill library changed; saved pools/cursor and compounds preserved')
    scaffold = selection['scaffold_smiles'] if selection else previous['scaffold_smiles']
    plan = _draw_plan(scaffold, library, fingerprint, rejected,
                      library_path, previous['batch_size'], maximum)
    for key in ('base_scaffold_smiles', 'attachment_selection', 'attachment_revision'):
        if key in previous:
            plan[key] = previous[key]
    if selection:
        plan.update(base_scaffold_smiles=original['scaffold_smiles'], attachment_selection=selection,
                    attachment_revision=previous.get('attachment_revision', 0) + 1)
    return database.advance_combichem_plan(args.db_path, iteration, previous, plan, args.target_count,
                                           change_attachments=change_attachments)


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
