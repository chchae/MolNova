"""Scaffold-related public helpers."""
from molnova import _core
from molnova import database
from pathlib import Path
import json
import os
import tempfile
import re
from rdkit import Chem

infer_mcs_smarts = _core.infer_mcs_smarts_from_iteration0
prepare_libinvent_scaffold = _core.prepare_libinvent_scaffold


def _publish(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _legacy_scaffold(args):
    mcs_file = args.output / 'mcs.smarts'
    scaffold_file = args.output / 'libinvent_scaffold.smi'
    if not mcs_file.is_file() or not scaffold_file.is_file():
        return None
    return dict(mcs_smarts=mcs_file.read_text().strip(),
                scaffold_smiles=scaffold_file.read_text().strip(),
                source='legacy_project_scaffold', source_ids=[])


def _validate(snapshot):
    query = Chem.MolFromSmarts(snapshot['mcs_smarts'])
    scaffold = Chem.MolFromSmiles(snapshot['scaffold_smiles'])
    if query is None or query.GetNumHeavyAtoms() == 0 or scaffold is None:
        raise RuntimeError('Invalid saved MCS/scaffold.')
    return query, scaffold


def _initial_scaffold(args):
    # Row zero stores the fixed initial constraint, not an elite-derived core.
    initial = database.iteration_scaffold(args.db_path, 0)
    if initial is None:
        initial = database.iteration_scaffold(args.db_path, 1) or _legacy_scaffold(args)
        if initial is None:
            references = _core.get_reference_compounds(args.db_path)
            supplied = getattr(args, 'initial_mcs_smarts', None)
            core = supplied if supplied is not None else infer_mcs_smarts(references, timeout=args.mcs_timeout)
            initial = dict(mcs_smarts=core,
                           source='user_initial_mcs' if supplied else 'iteration0_references',
                           source_ids=[ref['id'] for ref in references])
        initial = dict(initial)
        explicit_elite = getattr(args, 'elite_mcs_smarts', None)
        initial['initial_elite_mcs'] = (
            explicit_elite if explicit_elite is not None else
            initial['mcs_smarts'] if getattr(args, 'initial_mcs_smarts', None) is not None else None
        )
    # A docking SMARTS needs no LibInvent attachment points or SMILES conversion.
    query = Chem.MolFromSmarts(initial['mcs_smarts'])
    if query is None or query.GetNumHeavyAtoms() == 0:
        raise RuntimeError('Invalid initial docking MCS.')
    if not any(ref['mol'].HasSubstructMatch(query, useChirality=False)
               for ref in _core.get_reference_compounds(args.db_path)):
        raise RuntimeError('Initial docking core matches no original reference pose.')
    initial = database.iteration_scaffold(args.db_path, 0, initial)
    _publish(args.output / 'docking_mcs.smarts', initial['mcs_smarts'] + '\n')
    return initial


def mcs_display_summary(smarts):
    """Relax explicit RDKit ring predicates for display, never for calculations."""
    summary = re.sub(r'&!?R(?=[\]&])', '', smarts)
    summary = re.sub(r'([-:=#~])&!?@', r'\1', summary)
    if summary != smarts and Chem.MolFromSmarts(summary) is not None:
        return summary
    return smarts


def _previous_scaffold_fallback(args, iteration, inputs, docking_smarts, error):
    """Reuse only N-1's saved scaffold, after checking every selected elite."""
    previous = database.iteration_scaffold(args.db_path, iteration - 1)
    try:
        if previous is None:
            raise RuntimeError(f'iteration {iteration - 1} has no saved scaffold')
        query, scaffold = _validate(previous)
        dummies = [atom for atom in scaffold.GetAtoms() if atom.GetAtomicNum() == 0]
        combichem = getattr(args, 'sidechain_library', None) is not None
        expected = list(range(len(dummies))) if combichem else [0, 1]
        if (not dummies or (not combichem and len(dummies) != 2)
                or sorted(atom.GetAtomMapNum() for atom in dummies) != expected
                or any(atom.GetDegree() != 1 or
                       atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE for atom in dummies)):
            raise RuntimeError('saved scaffold needs labelled terminal single-bond attachments' if combichem else
                               'saved scaffold needs exactly two terminal single-bond attachments [*:0]/[*:1]')
        if not scaffold.HasSubstructMatch(query, useChirality=False):
            raise RuntimeError('saved scaffold does not contain its saved generation core')
        docking_query = Chem.MolFromSmarts(docking_smarts)
        if docking_query is None or not scaffold.HasSubstructMatch(docking_query, useChirality=False):
            raise RuntimeError('saved scaffold does not contain the fixed docking core')
        incompatible = [ref['id'] for ref in inputs
                        if _core.decompose_elite_for_libinvent(
                            Chem.MolToSmiles(ref['mol'], isomericSmiles=True),
                            previous['scaffold_smiles']) is None]
        if incompatible:
            raise RuntimeError(f'saved scaffold cannot decompose selected elite IDs {incompatible}')
    except (KeyError, ValueError, RuntimeError) as validation_error:
        raise RuntimeError(
            f'{error} Iteration {iteration}: previous-scaffold fallback rejected: {validation_error}.'
        ) from error
    return {
        'mcs_smarts': previous['mcs_smarts'],
        'scaffold_smiles': previous['scaffold_smiles'],
        'source': 'previous_iteration_scaffold_fallback',
        'source_ids': [ref['id'] for ref in inputs],
        'fallback_from_iteration': iteration - 1,
        'fallback_reason': 'insufficient_attachment_sites',
        'attempted_mcs_smarts': error.mcs_smarts,
    }


def log_iteration_mcs(args, iteration, snapshot=None, *, include_generation=True):
    """Display recorded cores without calculating or changing scientific work."""
    if not getattr(args, 'iteration_mcs', getattr(args, 'automatic_mcs', False)):
        return
    if snapshot is None:
        snapshot = database.iteration_scaffold(args.db_path, iteration)
    initial = (database.iteration_scaffold(args.db_path, 0)
               or database.iteration_scaffold(args.db_path, 1)
               or _legacy_scaffold(args))
    initial_smarts = (initial['mcs_smarts'] if initial
                      else getattr(args, 'initial_mcs_smarts', None))
    if initial_smarts:
        print(f'[MCS] Initial docking constraint: {initial_smarts}', flush=True)
    if snapshot and include_generation:
        source = snapshot['source']
        if source == 'previous_iteration_elites':
            source = f'iteration {iteration - 1} elites'
        elif source == 'user_initial_mcs':
            source = 'user supplied'
        elif source == 'user_elite_mcs':
            source = 'saved elite-mcs'
            current = getattr(args, 'elite_mcs_smarts', None)
            if current is None:
                source += '; absent from current TOML'
            elif current != snapshot.get('supplied_elite_mcs', snapshot['mcs_smarts']):
                source += '; current TOML differs'
        elif source == 'fixed_elite_mcs':
            source = f"fixed elite-mcs from iteration {snapshot['fixed_from_iteration']}"
        elif source == 'reference_mcs_fallback':
            source = 'reference-mcs fallback for generation'
        elif source == 'iteration0_references':
            source = 'initial references'
        elif source == 'legacy_project_scaffold':
            source = 'saved legacy scaffold'
        elif source == 'previous_iteration_scaffold_fallback':
            source = f"previous scaffold fallback from iteration {snapshot['fallback_from_iteration']}"
        elif source == 'single_elite_sidechain_expansion':
            source = f"single elite sidechain C-H expansion from iteration {snapshot['expansion_from_iteration']}"
        elif source == 'single_elite_sidechain_replacement':
            source = f"single elite terminal sidechain replacement from iteration {snapshot['replacement_from_iteration']}"
        if snapshot.get('fallback_reason') == 'insufficient_attachment_sites':
            problem = ('no attachment sites' if snapshot.get('generation_method') == 'combichem' else
                       'fewer than two attachment sites')
            print(f"[MCS] Iteration {iteration}: automatic MCS had {problem}; "
                  f"reusing validated scaffold from iteration {snapshot['fallback_from_iteration']}.", flush=True)
        summary = mcs_display_summary(snapshot['mcs_smarts'])
        label = 'generation core summary' if summary != snapshot['mcs_smarts'] else 'generation core'
        print(f'[MCS] Iteration {iteration} {label} ({source}): '
              f'{summary}', flush=True)
        if snapshot.get('scaffold_smiles'):
            print(f'[MCS] Iteration {iteration} generation scaffold SMILES: '
                  f"{snapshot['scaffold_smiles']}", flush=True)
    if snapshot:
        docking_smarts = snapshot.get('docking_mcs_smarts', snapshot['mcs_smarts'])
        if docking_smarts != initial_smarts:
            print(f'[MCS] Iteration {iteration} preserved Glide constraint: '
                  f'{docking_smarts}', flush=True)


def prepare_iteration_scaffold(args, iteration, elites=None, *, create=False, include_generation=True):
    """Adapt generation to elites while keeping the initial docking constraint."""
    if not getattr(args, 'iteration_mcs', getattr(args, 'automatic_mcs', False)):
        return

    snapshot = database.iteration_scaffold(args.db_path, iteration)
    method = 'combichem' if getattr(args, 'sidechain_library', None) is not None else 'reinvent4'
    if snapshot is None and database.iteration_has_compounds(args.db_path, iteration):
        # Pre-upgrade iterations already used the global core. Never change it
        # underneath generated compounds, prepared structures or submitted jobs.
        snapshot = _legacy_scaffold(args)
        if snapshot is None:
            raise RuntimeError(
                f'Iteration {iteration}: existing compounds have no saved scaffold; '
                'restore the original mcs.smarts and libinvent_scaffold.smi before resuming.'
            )
    if create and snapshot is not None and snapshot.get('generation_method', 'reinvent4') != method:
        raise RuntimeError(f'Iteration {iteration}: cannot switch generator for a saved iteration; '
                           'restore its original generation configuration.')

    if snapshot is None:
        if not create:
            raise RuntimeError(f'Iteration {iteration}: no saved generation scaffold.')
        initial = _initial_scaffold(args)
        fixed = getattr(args, 'elite_mcs_fixed', False)
        explicit_elite = getattr(args, 'elite_mcs_smarts', None)
        fallback = initial.get('initial_elite_mcs')
        if 'initial_elite_mcs' not in initial and getattr(args, 'initial_mcs_smarts', None) is not None:
            # Older row-zero snapshots already define the original reference core.
            fallback = initial['mcs_smarts']
        if not fixed and database.prior_generation_exists(args.db_path, iteration):
            # Bootstrap fallback belongs to the project's first generation,
            # including legacy iterations without the new application marker.
            fallback = None
        supplied = explicit_elite if explicit_elite is not None else fallback
        use_supplied = supplied is not None and (fixed or not database.elite_mcs_was_used(args.db_path, supplied))
        pinned = database.fixed_generation_scaffold(args.db_path, iteration, supplied) if fixed else None
        if iteration == 1:
            inputs = _core.get_reference_compounds(args.db_path)
        else:
            if elites is None or len(elites) != args.gbsa_elite_count:
                raise RuntimeError('Automatic MCS requires the selected preceding-iteration elites.')
            if any(row[3] != iteration - 1 for row in elites):
                raise RuntimeError('Automatic MCS inputs must belong to the preceding iteration.')
            inputs = [dict(id=row[0], name=row[1], mol=Chem.MolFromSmiles(row[2]))
                      for row in elites]
        source = (('user_elite_mcs' if explicit_elite is not None else 'reference_mcs_fallback') if use_supplied else
                  'iteration0_references' if iteration == 1 else 'previous_iteration_elites')
        if any(ref['mol'] is None for ref in inputs):
            raise RuntimeError('Invalid molecule in automatic MCS inputs.')
        if pinned:
            original_iteration, original = pinned
            snapshot = dict(original)
            snapshot['fixed_from_iteration'] = original.get('fixed_from_iteration', original_iteration)
        elif method == 'combichem' and iteration > 1 and len(inputs) == 1 and not use_supplied:
            from molnova.chemistry.combichem import prepare_single_elite_scaffold
            previous = (database.combichem_compound_scaffold(args.db_path, inputs[0]['id'])
                        or database.iteration_scaffold(args.db_path, iteration - 1))
            snapshot = prepare_single_elite_scaffold(
                inputs[0], previous, initial['mcs_smarts'],
            )
            snapshot['replacement_from_iteration'] = iteration - 1
            if fixed:
                snapshot['fixed_from_iteration'] = iteration
        else:
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    if method == 'combichem':
                        from molnova.chemistry.combichem import prepare_scaffold
                        info = prepare_scaffold(inputs, supplied if use_supplied else None,
                                                mcs_timeout=args.mcs_timeout)
                    else:
                        info = _core.prepare_libinvent_scaffold(
                            inputs, supplied if use_supplied else None, Path(temporary), mcs_timeout=args.mcs_timeout,
                        )
            except _core.InsufficientAttachmentSitesError as error:
                if use_supplied or iteration == 1:
                    raise
                snapshot = _previous_scaffold_fallback(
                    args, iteration, inputs, initial['mcs_smarts'], error,
                )
            else:
                snapshot = {
                    'mcs_smarts': info['mcs_smarts'], 'scaffold_smiles': info['scaffold_smiles'],
                    'source': source, 'source_ids': [ref['id'] for ref in inputs],
                }
            if use_supplied:
                snapshot['supplied_elite_mcs'] = supplied
            if fixed:
                snapshot['fixed_from_iteration'] = iteration
        if fixed:
            snapshot['source'] = 'fixed_elite_mcs'
            snapshot['elite_mcs_fixed'] = True
        snapshot['docking_mcs_smarts'] = initial['mcs_smarts']
        snapshot['generation_method'] = method

    _, scaffold = _validate(snapshot)
    if snapshot.get('generation_method') == 'combichem':
        from molnova.chemistry.combichem import attachment_atoms
        attachment_atoms(scaffold)
    elif create and snapshot.get('generation_method') == 'reinvent4':
        if sum(atom.GetAtomicNum() == 0 for atom in scaffold.GetAtoms()) != 2:
            raise RuntimeError('REINVENT4 generation requires exactly two scaffold attachments')
    # Old iterations used their generation core for docking. Preserve that
    # exact constraint beneath existing structures/jobs; only new cycles split it.
    docking_smarts = snapshot.get('docking_mcs_smarts', snapshot['mcs_smarts'])
    docking_query = Chem.MolFromSmarts(docking_smarts)
    if docking_query is None or docking_query.GetNumHeavyAtoms() == 0:
        raise RuntimeError(f'Iteration {iteration}: invalid docking constraint.')
    if not scaffold.HasSubstructMatch(docking_query, useChirality=False):
        raise RuntimeError(f'Iteration {iteration}: generation scaffold does not contain the fixed docking core.')
    if not any(ref['mol'].HasSubstructMatch(docking_query, useChirality=False)
               for ref in _core.get_reference_compounds(args.db_path)):
        raise RuntimeError(
            f'Iteration {iteration}: docking core matches no original reference pose; '
            'provide a compatible reference pose or an explicit reference-mcs.'
        )
    snapshot = database.iteration_scaffold(args.db_path, iteration, snapshot)
    directory = args.output / f'iter{iteration}' / 'scaffold'
    _publish(directory / 'mcs.smarts', snapshot['mcs_smarts'] + '\n')
    _publish(directory / 'libinvent_scaffold.smi', snapshot['scaffold_smiles'] + '\n')
    _publish(directory / 'scaffold_info.json', json.dumps(snapshot, indent=2) + '\n')
    _publish(directory / 'docking_mcs.smarts', docking_smarts + '\n')
    args.generation_mcs_smarts = snapshot['mcs_smarts']
    args.mcs_smarts = docking_smarts
    args.libinvent_scaffold_smiles = snapshot['scaffold_smiles']
    args.libinvent_scaffold_file = directory / 'libinvent_scaffold.smi'
    log_iteration_mcs(args, iteration, snapshot, include_generation=include_generation)
