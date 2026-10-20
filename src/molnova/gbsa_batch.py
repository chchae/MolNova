"""Native Prime/JobDJ batching; individual queues remain recovery-only."""
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path

from molnova import _core as core, database, prime_async as prime
from molnova.config import mmgbsa_job_options, prime_flexdist
from molnova.gbsa_streaming import collect_completed_subjobs
from molnova.schrodinger_guard import submission_guard, SubmissionBlocked
from molnova.stages._logging import format_elapsed

MODE = 'native_batch_v1'


def other_submission(args, directory):
    return next((path for path in Path(args.output).glob('iter*/mmgbsa/prime_job.json')
                 if path.parent.resolve() != Path(directory).resolve()), None)


def create_batch(args, directory, compound_ids):
    """Freeze the exact native request before any submission can occur."""
    directory = Path(directory)
    if prime.load_job(directory) is not None:
        raise RuntimeError('An existing Prime submission requires recovery')
    if other_submission(args, directory) is not None:
        raise SubmissionBlocked('Another project iteration requires Prime recovery')
    command = [args.schrodinger / 'prime_mmgbsa', directory / 'mmgbsa_input.maegz',
               '-OVERWRITE', '-report_prime_log', 'yes',
               *mmgbsa_job_options(args.host, len(compound_ids))]
    flexibility = prime_flexdist({'prime-flexdist': getattr(args, 'prime_flexdist', 0.0)})
    if flexibility:
        prepare_complexes(args, directory, compound_ids)
        command[1] = directory / 'mmgbsa_complexes.maegz'
        command += ['-ligand', 'atom.i_molnova_batch_ligand 1']
        command += ['-rflexdist', str(flexibility), '-rflexgroup', 'side',
                    '-out_type', 'COMPLEX']
    record = dict(mode=MODE, command=[str(value) for value in command],
                  compound_ids=list(compound_ids), prime_flexdist=flexibility,
                  attempt=0, job_id=None, launching=False, retry_at=0,
                  started_at=time.time())
    prime._save(directory, record)
    return record


def prepare_complexes(args, directory, compound_ids):
    """Keep flexible selections/free receptors independent for each ligand.

    Native PV batching selects the union of receptor regions over all ligands.
    Native complex-input mode instead initializes a receptor for each complex.
    """
    script = directory / '_prepare_mmgbsa_complexes.py'
    script.write_text('''import json, re, sys
from schrodinger import structure
wanted = set(json.loads(sys.argv[3]))
seen = set()
reader = iter(structure.StructureReader(sys.argv[1]))
receptor = next(reader)
for atom in receptor.atom:
    atom.property['i_molnova_batch_ligand'] = 0
with structure.StructureWriter(sys.argv[2]) as writer:
    for ligand in reader:
        match = re.fullmatch(r'CMPID_(\\d+)', ligand.title or '')
        cid = int(match.group(1)) if match else ligand.property.get('i_user_compound_id')
        if cid not in wanted or cid in seen:
            raise ValueError('Unexpected or duplicated complex input: %s' % cid)
        for atom in ligand.atom:
            atom.property['i_molnova_batch_ligand'] = 1
        complex_st = receptor.merge(ligand, copy_props=True)
        for key, value in ligand.property.items():
            complex_st.property[key] = value
        complex_st.title = ligand.title
        complex_st.property['i_user_compound_id'] = cid
        writer.append(complex_st)
        seen.add(cid)
if seen != wanted:
    raise ValueError('Missing complex inputs: %s' % sorted(wanted - seen))
''')
    pending = directory / 'complexes.pending.maegz'
    core.run_command([args.schrodinger / 'run', script, directory / 'mmgbsa_input.maegz',
                      pending, json.dumps(list(compound_ids))], cwd=directory)
    pending.replace(directory / 'mmgbsa_complexes.maegz')


@contextmanager
def submission_fence(args, directory):
    # Workers hold the project MM-GBSA stage lock. Reserve all seven slots for
    # one native parent; even an uncertain other manifest prevents admission.
    with submission_guard(args.schrodinger, 'mmgbsa'):
        path = other_submission(args, directory)
        if path is not None:
            raise SubmissionBlocked(f'Another project iteration requires Prime recovery: {path}')
        yield


def publish_coordinates(args, directory, compound_ids=()):
    """Merge native results, retaining coordinates from earlier scored batches."""
    directory = Path(directory)
    # Previous final snapshots come first: existing scores and their structures
    # must retain priority over outputs from a subsequent partial-batch retry.
    sources = sorted(directory.glob('previous-*/mmgbsa_final.maegz'))
    sources += sorted(directory.glob('previous-*/*-out.maegz'))
    sources += sorted(directory.glob('subjobs/*/*/result.maegz'))
    sources += sorted(directory.glob('*-out.maegz'))
    final = directory / 'mmgbsa_final.maegz'
    if final.exists():
        sources.insert(0, final)
    if not sources:
        return
    iteration = int(directory.parent.name.removeprefix('iter'))
    # Keep all SQL in database.py; this query also rejects nonfinite scores.
    ids = database.mmgbsa_scored_ids(args.db_path, iteration)
    ids_file = directory / 'final_ids.txt'
    ids_file.write_text('\n'.join(map(str, ids)) + '\n')
    required = sorted(set(ids) & set(compound_ids))
    script = directory / '_merge_mmgbsa_batch.py'
    script.write_text('''import re, sys
from schrodinger import structure
wanted = {int(line) for line in open(sys.argv[2]) if line.strip()}
import json
required = set(json.loads(sys.argv[3]))
poses = {}
receptor = None
for filename in sys.argv[4:]:
    for st in structure.StructureReader(filename):
        match = re.search(r'CMPID_(\\d+)', st.title or '')
        cid = int(match.group(1)) if match else st.property.get('i_user_compound_id')
        if cid is None and receptor is None:
            receptor = st
        elif cid in wanted and cid not in poses:
            poses[cid] = st
if required - poses.keys():
    raise ValueError('Missing scored MM-GBSA coordinates: %s' % sorted(required - poses.keys()))
with structure.StructureWriter(sys.argv[1]) as writer:
    if receptor is not None:
        writer.append(receptor)
    for cid in sorted(poses):
        writer.append(poses[cid])
''')
    pending = directory / 'mmgbsa_final.pending.maegz'
    try:
        core.run_command([args.schrodinger / 'run', script, pending, ids_file,
                          json.dumps(required), *sources],
                         cwd=directory)
        if not pending.is_file():
            raise RuntimeError('Merged MM-GBSA coordinates were not created')
        pending.replace(final)
    finally:
        pending.unlink(missing_ok=True)


def poll_batch(iteration, args, directory, record):
    def collect(saved, jobs):
        collect_completed_subjobs(args, iteration, directory, saved, jobs)
        parent = next(job for job in jobs if job['jobId'] == saved['job_id'])
        if parent['status'] != 'DONE' and all(
                job.get('status') in {'DONE', 'FAILED', 'CANCELED', 'STOPPED'} for job in jobs):
            stem = Path(saved['command'][1]).stem
            outstanding = [job['jobId'] for job in jobs
                           if job.get('parentJobId') == saved['job_id']
                           and job.get('status') == 'DONE'
                           and re.fullmatch(re.escape(stem) + r'\.\d+', job.get('jobName', ''))
                           and job['jobId'] not in saved.get('imported_subjobs', [])]
            if outstanding:
                raise prime.PrimeReconciliationPending(
                    'Terminal Prime subjob outputs still require collection: ' + ', '.join(outstanding))
            publish_coordinates(args, directory, saved['compound_ids'])

    if not prime.submit_or_poll(
            record['command'], directory, record['compound_ids'],
            wait_seconds=getattr(args, 'gbsa_license_retry_seconds', 300),
            retries=getattr(args, 'gbsa_license_retries', 3),
            on_completed_subjobs=collect,
            submission_fence=lambda: submission_fence(args, directory)):
        running, saved = database.mmgbsa_progress_counts(args.db_path, iteration)
        print(f'{iteration:<3d}: GBSA running={running:<3d}, saved={saved:<3d}, '
              'scheduler=Prime, limit=7.', flush=True)
        return None
    updated = core.finish_iteration_mmgbsa(iteration, args, directory, record['compound_ids'])
    prime.persist_atomtyping_failures(directory, args.db_path)
    # Keep the live JobId until final coordinates publish successfully. A failed
    # merge is retried after restart without launching a second calculation.
    publish_coordinates(args, directory, record['compound_ids'])
    current = prime.load_job(directory)
    current['finished_at'] = time.time()
    prime._save(directory, current)
    prime.job_file(directory).replace(Path(directory) / f"batch-{time.time_ns()}.completed.json")
    print(f'Iteration {iteration}: Prime batch completed; total elapsed='
          f"{format_elapsed(current['finished_at'] - current['started_at'])}.", flush=True)
    return updated
