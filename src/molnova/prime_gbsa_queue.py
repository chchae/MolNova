"""Durable per-compound Prime queue, refilled independently of JobDJ."""
import json
import shutil
import time
from pathlib import Path
from molnova.paths import stage_manifests
from molnova.stages._logging import detail

from molnova import _core as c, database
from molnova import prime_gbsa_async as prime
from molnova.schrodinger_guard import submission_guard, SubmissionBlocked
from molnova.states import CompoundState as State
from molnova.config import prime_gbsa_flexdist
from molnova.stages._logging import format_elapsed

MAX_ACTIVE = 7
MODE = 'compound_queue_v1'


def prepare_new_calculation(args, directory):
    """Remove stale working files before building a new PV; preserve scored history."""
    directory = Path(directory)
    if prime.load_job(directory) is not None or any(directory.glob('queue-*/CMPID_*/prime_gbsa_job.json')):
        raise RuntimeError('prime_gbsa submissions still require recovery; working files retained.')
    from molnova.schrodinger_guard import active_jobs, belongs_to_directory
    if any(directory.iterdir()) and any(
            belongs_to_directory(job, directory) for job in active_jobs(args.schrodinger)):
        raise RuntimeError('prime_gbsa native jobs still running; working files retained.')
    source = getattr(args, 'prime_gbsa_receptor', None)
    keep = Path(source).resolve() if source is not None else None
    files = [path for path in directory.iterdir() if path.is_file()
             and not path.name.endswith('.completed.json') and path.resolve() != keep]
    if files:
        archive = directory / f'previous-{time.time_ns()}'
        archive.mkdir()
        for path in files:
            path.replace(archive / path.name)
        print(f'Cleared {len(files)} previous prime_gbsa working files; building fresh inputs.', flush=True)


def prepare_inputs(args, directory, record):
    """Split the validated PV into receptor + exactly one original ligand."""
    script = directory / '_split_prime_gbsa_queue.py'
    script.write_text(r'''
import json, re, sys
from pathlib import Path
from schrodinger import structure
root = Path(sys.argv[1])
record = json.loads((root / 'prime_gbsa_job.json').read_text())
reader = iter(structure.StructureReader(str(root / 'prime_gbsa_input.maegz')))
receptor = next(reader)
wanted = set(record['compound_ids'])
seen = set()
for st in reader:
    match = re.fullmatch(r'CMPID_(\d+)', st.title or '')
    cid = int(match.group(1)) if match else st.property.get('i_user_compound_id')
    if cid not in wanted:
        raise ValueError('Unexpected prime_gbsa input compound: %s' % cid)
    if cid in seen:
        raise ValueError('Multiple best poses for compound %s' % cid)
    seen.add(cid)
    folder = root / record['run_dir'] / ('CMPID_%s' % cid)
    folder.mkdir(parents=True, exist_ok=True)
    pending = folder / 'input.pending.maegz'
    with structure.StructureWriter(str(pending)) as writer:
        writer.append(receptor)
        writer.append(st)
    pending.replace(folder / 'prime_gbsa.maegz')
if seen != wanted:
    raise ValueError('Missing best poses for compounds: %s' % sorted(wanted - seen))
print('Prepared single-compound prime_gbsa inputs:', len(seen))
'''.lstrip())
    c.run_command([args.schrodinger / 'run', script, directory], cwd=directory)
    record['prepared'] = True
    prime._save(directory, record)


def create_queue(args, directory, compound_ids):
    flexibility = prime_gbsa_flexdist({'prime-gbsa-flexdist': getattr(args, 'prime_gbsa_flexdist', 0.0)})
    started_ns = time.time_ns()
    record = {'mode': MODE, 'compound_ids': list(compound_ids),
              'run_dir': f'queue-{started_ns}', 'started_at': started_ns / 1e9,
              'prepared': False,
              'prime_gbsa_flexdist': flexibility,
              'entries': {str(cid): 'pending' for cid in compound_ids}}
    # Reserve the entire selected set before preparing inputs or launching jobs.
    prime._save(directory, record)
    if flexibility:
        print(f'Prime receptor flexibility: residues within {flexibility:g} Angstrom of each ligand.', flush=True)
    return record


def _folder(directory, record, cid):
    return directory / record['run_dir'] / f'CMPID_{cid}'


def other_active_slots(args, directory):
    """The seven-slot ceiling applies across all project iterations."""
    active = 0
    for path in stage_manifests(args.output, 'prime_gbsa', 'prime_gbsa_job.json'):
        if path.parent == directory:
            continue
        other = json.loads(path.read_text())
        if other.get('mode') != MODE:
            # A legacy batch owns its native resource request until recovered.
            return MAX_ACTIVE
        active += sum(state == 'active' for state in other['entries'].values())
    return active


def merge_final_coordinates(args, directory, record):
    """Publish completed native structures without changing coordinates/properties."""
    records = [json.loads(path.read_text())
               for path in sorted(directory.glob('queue-*.completed.json'))]
    records.append(record)
    sources = {}
    for completed in records:
        if completed.get('mode') != MODE:
            continue
        for cid in completed['compound_ids']:
            if completed['entries'][str(cid)] != 'done' or cid in sources:
                continue
            folder = _folder(directory, completed, cid)
            outputs = sorted(folder.glob('*-out.maegz'),
                             key=lambda path: path.stat().st_mtime, reverse=True)
            if not outputs:
                raise RuntimeError(f'CMPID_{cid}: final prime_gbsa coordinates are missing in {folder}')
            sources[cid] = outputs[0]
    if not sources:
        return None
    output = directory / 'prime_gbsa_final.maegz'
    pending = directory / 'prime_gbsa_final.pending.maegz'
    merger = c.create_best_pose_merger(directory)
    try:
        c.run_command([args.schrodinger / 'run', merger, pending,
                       *(sources[cid] for cid in sorted(sources))], cwd=directory,
                      log_message=f'prime_gbsa final coordinates: {len(sources)} results -> {output.name}')
        if not pending.is_file():
            raise RuntimeError('Merged prime_gbsa coordinates were not created')
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    return output


def poll_queue(iteration, args, directory, record):
    """Check active jobs, commit each result, then immediately refill free slots."""
    directory = Path(directory)
    # Pin the protocol for the entire queue, including not-yet-submitted members.
    # Missing metadata means the historical frozen-receptor protocol, never TOML.
    flexibility = prime_gbsa_flexdist({'prime-gbsa-flexdist': record.get('prime_gbsa_flexdist', 0.0)})
    if 'started_at' not in record:
        # Older queues already encode their creation timestamp in the run directory.
        # Recover it rather than restarting the timer at this worker invocation.
        record['started_at'] = int(record['run_dir'].removeprefix('queue-')) / 1e9
        prime._save(directory, record)
    if not record['prepared']:
        prepare_inputs(args, directory, record)
    updated = 0
    blocked = set()
    rows = database.prime_gbsa_submission_rows(args.db_path, record['compound_ids'])
    hosts = [entry.split(':', 1)[0] for entry in args.host.split()]
    if not hosts:
        raise ValueError('prime_gbsa requires a configured host')

    def advance(cid):
        nonlocal updated
        key = str(cid)
        folder = _folder(directory, record, cid)
        # A committed result alone cannot retire an uncertain external job.
        # Import/record cleanup must still recover its saved JobId.
        saved = prime.load_job(folder)
        if rows[cid][2] is not None and saved is None:
            record['entries'][key] = 'done'
            prime._save(directory, record)
            return
        command = [args.schrodinger / 'prime_mmgbsa', folder / 'prime_gbsa.maegz',
                   '-OVERWRITE', '-HOST', hosts[record['compound_ids'].index(cid) % len(hosts)],
                   '-NJOBS', '1']
        if flexibility:
            # PV output contains the original receptor, so request the native
            # optimized complex when receptor atoms are allowed to move.
            command += ['-rflexdist', str(flexibility), '-rflexgroup', 'residue', '-out_type', 'COMPLEX']
        if saved is not None and 'command' in saved:
            command = saved['command']
        record['entries'][key] = 'active'
        prime._save(directory, record)
        try:
            ready = prime.submit_or_poll(command, folder, [cid],
                wait_seconds=getattr(args, 'prime_gbsa_license_retry_seconds', 300),
                retries=getattr(args, 'prime_gbsa_license_retries', 3))
            if not ready:
                # A deferred license retry owns no external job/CPU slot.
                saved = prime.load_job(folder)
                if saved and not saved.get('job_id') and not saved.get('launching'):
                    record['entries'][key] = 'pending'
                    blocked.add(cid)
                    prime._save(directory, record)
                return
            changed = c.finish_iteration_prime_gbsa(iteration, args, folder, [cid])
            current = database.prime_gbsa_submission_rows(args.db_path, [cid])[cid]
            if current[2] is None:
                prime.job_file(folder).unlink(missing_ok=True)
                raise RuntimeError(f'CMPID_{cid}: completed prime_gbsa returned no score')
            updated += changed
            shutil.copy2(folder / 'prime_gbsa_top.tsv', directory / 'prime_gbsa_top.tsv')
            record['entries'][key] = 'done'
            # DB commit precedes the durable queue checkpoint.
            prime._save(directory, record)
            prime.job_file(folder).unlink(missing_ok=True)
            detail(f'Iteration {iteration}: CMPID_{cid} GBSA score persisted; slot released.', flush=True)
        except Exception as exc:
            if prime.load_job(folder) is not None:
                record['entries'][key] = 'active'
                # Status/transfer/unknown-launch errors never permit resubmission.
                print(f'Iteration {iteration}: CMPID_{cid} recovery pending: {exc}', flush=True)
                return
            if isinstance(exc, prime.PrimeAtomTypingError):
                database.mark_prime_gbsa_atomtyping_failures(args.db_path, [cid], str(exc))
            else:
                current = database.prime_gbsa_submission_rows(args.db_path, [cid])[cid]
                if current[2] is None:
                    c.set_compound_state([cid], State.FAILED, failed_stage='prime_gbsa',
                                         failure_message=str(exc), db_path=args.db_path)
            record['entries'][key] = 'failed'
            prime._save(directory, record)
            print(f'Iteration {iteration}: CMPID_{cid} GBSA failed: {exc}', flush=True)

    occupied = other_active_slots(args, directory)

    submission_blocked = False

    def refill():
        nonlocal submission_blocked
        if submission_blocked:
            return
        if not any(state == 'pending' for state in record['entries'].values()):
            return
        try:
            with submission_guard(args.schrodinger, 'prime_gbsa'):
                fill_slots()
        except SubmissionBlocked as exc:
            submission_blocked = True
            print(f'Iteration {iteration}: prime_gbsa submissions waiting: {exc}.', flush=True)

    def fill_slots():
        for cid in record['compound_ids']:
            if record['entries'][str(cid)] != 'pending' or cid in blocked:
                continue
            if occupied + sum(state == 'active' for state in record['entries'].values()) >= MAX_ACTIVE:
                break
            advance(cid)

    # Snapshot only existing active jobs. Import one result and refill its slot
    # before checking/importing other results, avoiding collection-induced idle time.
    active_before_poll = [cid for cid in record['compound_ids']
                          if record['entries'][str(cid)] == 'active']
    for cid in active_before_poll:
        advance(cid)
        refill()
    refill()
    active = sum(state == 'active' for state in record['entries'].values())
    pending = sum(state == 'pending' for state in record['entries'].values())
    running, saved = database.prime_gbsa_progress_counts(args.db_path, iteration)
    print(f'{iteration:<3d}: prime_gbsa running={running:<3d}, saved={saved:<3d}, '
          f'queue={active}, waiting={pending:<3d}, '
          f'limit={MAX_ACTIVE}, updated={updated}.', flush=True)
    if active or pending:
        return None
    # Keep the live manifest until coordinates publish successfully. On failure,
    # the next worker retries merging, without resubmitting scored compounds.
    merge_final_coordinates(args, directory, record)
    record.setdefault('finished_at', time.time())
    prime._save(directory, record)
    prime.job_file(directory).replace(directory / f"{record['run_dir']}.completed.json")
    elapsed = format_elapsed(record['finished_at'] - record['started_at'])
    failed = sum(state == 'failed' for state in record['entries'].values())
    print(f'Iteration {iteration}: prime_gbsa queue completed; total elapsed={elapsed} '
          f'(including waiting/recovery), failed={failed}.', flush=True)
    return updated
