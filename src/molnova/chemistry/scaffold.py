"""Scaffold-related public helpers."""
from molnova import _core
from molnova import database
from pathlib import Path
import json
import os
import tempfile
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


def log_iteration_mcs(args, iteration, snapshot=None):
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
    if snapshot:
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
        print(f'[MCS] Iteration {iteration} generation core ({source}): '
              f"{snapshot['mcs_smarts']}", flush=True)
        docking_smarts = snapshot.get('docking_mcs_smarts', snapshot['mcs_smarts'])
        if docking_smarts != initial_smarts:
            print(f'[MCS] Iteration {iteration} preserved Glide constraint: '
                  f'{docking_smarts}', flush=True)


def prepare_iteration_scaffold(args, iteration, elites=None, *, create=False):
    """Adapt generation to elites while keeping the initial docking constraint."""
    if not getattr(args, 'iteration_mcs', getattr(args, 'automatic_mcs', False)):
        return

    snapshot = database.iteration_scaffold(args.db_path, iteration)
    if snapshot is None and database.iteration_has_compounds(args.db_path, iteration):
        # Pre-upgrade iterations already used the global core. Never change it
        # underneath generated compounds, prepared structures or submitted jobs.
        snapshot = _legacy_scaffold(args)
        if snapshot is None:
            raise RuntimeError(
                f'Iteration {iteration}: existing compounds have no saved scaffold; '
                'restore the original mcs.smarts and libinvent_scaffold.smi before resuming.'
            )

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
        else:
            with tempfile.TemporaryDirectory() as temporary:
                info = _core.prepare_libinvent_scaffold(
                    inputs, supplied if use_supplied else None, Path(temporary), mcs_timeout=args.mcs_timeout,
                )
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

    _, scaffold = _validate(snapshot)
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
    log_iteration_mcs(args, iteration, snapshot)
