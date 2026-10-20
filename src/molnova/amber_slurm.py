"""Slurm admission/recovery around the unchanged single-node Amber worker."""
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import uuid

MODE = 'slurm_batches_v1'
SINGLE = 'slurm_single_v1'


def new_record(args, iteration, directory, ids):
    from molnova import amber_gbsa as amber
    from pathlib import PurePosixPath
    if not ids or not 1 <= args.amber_cpus <= 384:
        raise ValueError('Slurm Amber requires compounds and 1..384 CPUs')
    budget = min(args.amber_cpus, len(ids))
    count = (budget + 191) // 192
    parts = []
    for index in range(count):
        run_id = 'run-' + uuid.uuid4().hex
        parts.append({'run_id': run_id, 'compound_ids': ids[index::count],
                      'cpus': budget // count + (index < budget % count),
                      'prepared': False, 'mode': SINGLE, 'iteration': iteration,
                      'db_path': str(args.db_path.resolve()), 'settings': dict(args.amber_remote),
                      'charge_method': args.amber_charge_method,
                      'remote_dir': str(PurePosixPath(args.amber_remote['remote_work_dir']) / run_id)})
    record = {'mode': MODE, 'run_id': 'run-' + uuid.uuid4().hex, 'iteration': iteration,
              'db_path': str(args.db_path.resolve()), 'settings': dict(args.amber_remote),
              'charge_method': args.amber_charge_method, 'compound_ids': ids,
              'source_fingerprints': amber.source_fingerprints(args, iteration),
              'parts': parts, 'cpus': args.amber_cpus}
    amber.save(directory / 'amber_job.json', record)
    return record


def prepare_files(folder, record):
    from molnova import amber_gbsa as amber
    control = Path(__file__).with_name('amber_slurm_control.py')
    (folder / control.name).write_bytes(control.read_bytes())
    remote = shlex.quote(record['remote_dir'])
    body = f'exec srun --ntasks=1 --cpus-per-task={record["cpus"]} python worker.py run {remote}'
    (folder / 'amber.sbatch').write_text('#!/bin/bash\nset -e\nexec ' +
                                       amber.environment_command(record['settings'], body) + '\n')
    return [folder / control.name, folder / 'amber.sbatch']


def job_state(directory, record):
    from molnova import amber_gbsa as amber
    control = shlex.quote(record['remote_dir'] + '/amber_slurm_control.py')
    remote = shlex.quote(record['remote_dir'])
    # No Amber environment on the submission host; system Python only handles
    # journals and Slurm commands. Amber activation happens only on allocated nodes.
    if not record.get('slurm_job_id'):
        result = amber.ssh(record['settings'], f'python3 {control} submit {remote}',
                           capture_output=True, text=True)
        submitted = json.loads(result.stdout)
        if submitted.get('busy'):
            return {'active': True, 'state': 'SUBMITTING'}
        job_id = submitted.get('job_id', '')
        if not isinstance(job_id, str) or not job_id.isdigit():
            raise RuntimeError('Missing Slurm JobId; submission retained for reconciliation')
        record['slurm_job_id'] = job_id
        amber.save(directory / 'amber_job.json', record)
    response = amber.ssh(record['settings'], f'python3 {control} status {record["slurm_job_id"]}',
                          capture_output=True, text=True)
    return json.loads(response.stdout)


def poll(args, iteration, directory, record):
    from molnova import amber_gbsa as amber
    if (record['settings'] != args.amber_remote
            or record['charge_method'] != args.amber_charge_method
            or record['cpus'] != args.amber_cpus):
        raise RuntimeError('Slurm Amber settings changed; recover saved work with its original settings')
    if record['iteration'] != iteration or record['db_path'] != str(args.db_path.resolve()):
        raise RuntimeError('Saved Slurm Amber work belongs to another project/iteration')
    amber.verify_sources(args, iteration, record)
    pending, errors = False, []
    for part in record['parts']:
        folder = directory / part['run_id']
        folder.mkdir(exist_ok=True)
        completed = folder / (part['run_id'] + '.completed.json')
        if completed.exists():
            continue  # Includes a crash after child retirement, before parent retirement.
        child = amber.load_job(folder)
        if child is None:
            child = dict(part)
            amber.save(folder / 'amber_job.json', child)
        child_args = SimpleNamespace(**vars(args))
        child_args.amber_cpus = part['cpus']
        try:
            amber.verify_sources(args, iteration, record)
            if not amber.poll(child_args, iteration, folder, child):
                pending = True
        except Exception as exc:
            # Continue collecting/submitting the other shard; preserve every
            # failed/unknown identity, committed score and native lock.
            errors.append(exc)
    if errors:
        raise RuntimeError('Slurm Amber recovery pending: ' + '; '.join(map(str, errors))) from errors[0]
    if pending:
        return False
    record['finished'] = True
    amber.save(directory / (record['run_id'] + '.completed.json'), record)
    (directory / 'amber_job.json').unlink()
    return True
