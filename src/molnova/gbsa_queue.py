"""Durable per-compound Prime queue, refilled independently of JobDJ."""
import json
import shutil
import time
from pathlib import Path

from molnova import _core as c, database
from molnova import prime_async as prime
from molnova.schrodinger_guard import submission_guard, SubmissionBlocked
from molnova.states import CompoundState as State

MAX_ACTIVE = 7
MODE = 'compound_queue_v1'


def prepare_inputs(args, directory, record):
    """Split the validated PV into receptor + exactly one original ligand."""
    script = directory / '_split_mmgbsa_queue.py'
    script.write_text(r'''
import json, re, sys
from pathlib import Path
from schrodinger import structure
root = Path(sys.argv[1])
record = json.loads((root / 'prime_job.json').read_text())
reader = iter(structure.StructureReader(str(root / 'mmgbsa_input.maegz')))
receptor = next(reader)
wanted = set(record['compound_ids'])
seen = set()
for st in reader:
    match = re.fullmatch(r'CMPID_(\d+)', st.title or '')
    cid = int(match.group(1)) if match else st.property.get('i_user_compound_id')
    if cid not in wanted:
        raise ValueError('Unexpected MM-GBSA input compound: %s' % cid)
    if cid in seen:
        raise ValueError('Multiple best poses for compound %s' % cid)
    seen.add(cid)
    folder = root / record['run_dir'] / ('CMPID_%s' % cid)
    folder.mkdir(parents=True, exist_ok=True)
    pending = folder / 'input.pending.maegz'
    with structure.StructureWriter(str(pending)) as writer:
        writer.append(receptor)
        writer.append(st)
    pending.replace(folder / 'mmgbsa_input.maegz')
if seen != wanted:
    raise ValueError('Missing best poses for compounds: %s' % sorted(wanted - seen))
print('Prepared single-compound MM-GBSA inputs:', len(seen))
'''.lstrip())
    c.run_command([args.schrodinger / 'run', script, directory], cwd=directory)
    record['prepared'] = True
    prime._save(directory, record)


def create_queue(args, directory, compound_ids):
    record = {'mode': MODE, 'compound_ids': list(compound_ids),
              'run_dir': f'queue-{time.time_ns()}', 'prepared': False,
              'entries': {str(cid): 'pending' for cid in compound_ids}}
    # Reserve the entire selected set before preparing inputs or launching jobs.
    prime._save(directory, record)
    return record


def _folder(directory, record, cid):
    return directory / record['run_dir'] / f'CMPID_{cid}'


def other_active_slots(args, directory):
    """The seven-slot ceiling applies across all project iterations."""
    active = 0
    for path in args.output.glob('iter*/mmgbsa/prime_job.json'):
        if path.parent == directory:
            continue
        other = json.loads(path.read_text())
        if other.get('mode') != MODE:
            # A legacy batch owns its native resource request until recovered.
            return MAX_ACTIVE
        active += sum(state == 'active' for state in other['entries'].values())
    return active


def poll_queue(iteration, args, directory, record):
    """Check active jobs, commit each result, then immediately refill free slots."""
    directory = Path(directory)
    if not record['prepared']:
        prepare_inputs(args, directory, record)
    updated = 0
    blocked = set()
    rows = database.mmgbsa_submission_rows(args.db_path, record['compound_ids'])
    hosts = [entry.split(':', 1)[0] for entry in args.host.split()]
    if not hosts:
        raise ValueError('MM-GBSA requires a configured host')

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
        command = [args.schrodinger / 'prime_mmgbsa', folder / 'mmgbsa_input.maegz',
                   '-OVERWRITE', '-HOST', hosts[record['compound_ids'].index(cid) % len(hosts)],
                   '-NJOBS', '1']
        record['entries'][key] = 'active'
        prime._save(directory, record)
        try:
            ready = prime.submit_or_poll(command, folder, [cid],
                wait_seconds=getattr(args, 'gbsa_license_retry_seconds', 300),
                retries=getattr(args, 'gbsa_license_retries', 3))
            if not ready:
                # A deferred license retry owns no external job/CPU slot.
                saved = prime.load_job(folder)
                if saved and not saved.get('job_id') and not saved.get('launching'):
                    record['entries'][key] = 'pending'
                    blocked.add(cid)
                    prime._save(directory, record)
                return
            changed = c.finish_iteration_mmgbsa(iteration, args, folder, [cid])
            current = database.mmgbsa_submission_rows(args.db_path, [cid])[cid]
            if current[2] is None:
                prime.job_file(folder).unlink(missing_ok=True)
                raise RuntimeError(f'CMPID_{cid}: completed MM-GBSA returned no score')
            updated += changed
            shutil.copy2(folder / 'gbsa_top.tsv', directory / 'gbsa_top.tsv')
            record['entries'][key] = 'done'
            # DB commit precedes the durable queue checkpoint.
            prime._save(directory, record)
            prime.job_file(folder).unlink(missing_ok=True)
            print(f'Iteration {iteration}: CMPID_{cid} GBSA score persisted; slot released.', flush=True)
        except Exception as exc:
            if prime.load_job(folder) is not None:
                record['entries'][key] = 'active'
                # Status/transfer/unknown-launch errors never permit resubmission.
                print(f'Iteration {iteration}: CMPID_{cid} recovery pending: {exc}', flush=True)
                return
            if isinstance(exc, prime.PrimeAtomTypingError):
                database.mark_gbsa_atomtyping_failures(args.db_path, [cid], str(exc))
            else:
                current = database.mmgbsa_submission_rows(args.db_path, [cid])[cid]
                if current[2] is None:
                    c.set_compound_state([cid], State.FAILED, failed_stage='gbsa',
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
            with submission_guard(args.schrodinger, 'mmgbsa'):
                fill_slots()
        except SubmissionBlocked as exc:
            submission_blocked = True
            print(f'Iteration {iteration}: MM-GBSA submissions waiting: {exc}.', flush=True)

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
    print(f'Iteration {iteration}: MM-GBSA queue active={active}, waiting={pending}, '
          f'limit={MAX_ACTIVE}, scores updated={updated}.', flush=True)
    if active or pending:
        return None
    prime.job_file(directory).replace(directory / f"{record['run_dir']}.completed.json")
    return updated
