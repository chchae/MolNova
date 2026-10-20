"""Standalone Slurm submission journal, copied to the remote run directory.

No Amber activation or scientific computation on the Slurm submission host.
"""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}


def save(path, record):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(record))
    pending.replace(path)


def query(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=30)


def recover(record):
    """A unique job name reconciles loss of the sbatch response, including DONE."""
    name = record['job_name']
    ids = {line.strip() for line in query(
        ['squeue', '--noheader', '--name', name, '--format=%A']).splitlines() if line.strip()}
    since = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(record['started_at'] - 86400))
    rows = query(['sacct', '--noheader', '--parsable2', '--starttime', since,
                  '--name', name, '--format=JobIDRaw,JobName%128'])
    for line in rows.splitlines():
        fields = line.split('|')
        if len(fields) >= 2 and fields[1].strip() == name and fields[0].strip().isdigit():
            ids.add(fields[0].strip())
    if any(not value.isdigit() for value in ids) or len(ids) > 1:
        raise RuntimeError('Ambiguous Slurm submission history; automatic resubmission blocked')
    return next(iter(ids), None)


def submit(root):
    with (root / 'slurm-submit.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'busy': True}
        path = root / 'slurm-submission.json'
        record = json.loads(path.read_text()) if path.exists() else {}
        if record.get('job_id'):
            return record
        if record.get('launching'):
            job_id = recover(record)  # A failed lookup is never proof of absence.
            if job_id:
                record.update(job_id=job_id, launching=False)
                save(path, record)
                return record
            if time.time() - record['started_at'] < 60:
                return {'busy': True}
            # sbatch inherits this lock. Acquiring it proves no prior submit
            # client survives; both active and terminal history were checked.
            save(root / f'slurm-unsubmitted-{time.time_ns()}.json', record)
        config = json.loads((root / 'input.json').read_text())
        options = config['slurm']
        cpus = config['cpus']
        if isinstance(cpus, bool) or not isinstance(cpus, int) or not 1 <= cpus <= 192:
            raise ValueError('Slurm Amber shard requires 1..192 CPUs')
        record = {'job_name': options['job_name'], 'started_at': time.time(),
                  'launching': True, 'job_id': None}
        save(path, record)
        command = ['sbatch', '--parsable', '--job-name', record['job_name'],
                   '--partition', options['partition'], '--nodes=1', '--ntasks=1',
                   '--cpus-per-task', str(cpus), '--hint=nomultithread', '--mem=0',
                   '--chdir', str(root), '--output', str(root / 'slurm-%j.out'),
                   '--error', str(root / 'slurm-%j.err'), str(root / 'amber.sbatch')]
        # Preserve the journal/uncertainty even on timeout or interrupted SSH.
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE,
                                         timeout=90, pass_fds=(lock.fileno(),))
        if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?\s*', output):
            raise RuntimeError('sbatch returned no unambiguous JobId; retained for reconciliation')
        record.update(job_id=output.strip().split(';')[0], launching=False)
        save(path, record)
        return record


def status(job_id):
    if not str(job_id).isdigit():
        raise ValueError('Invalid Slurm JobId')
    queued = query(['squeue', '--noheader', '--jobs', str(job_id), '--format=%T']).strip()
    if queued:
        return {'active': True, 'state': queued, 'job_id': str(job_id)}
    rows = query(['sacct', '--noheader', '--parsable2', '--jobs', str(job_id),
                  '--format=JobIDRaw,State'])
    states = {}
    for line in rows.splitlines():
        fields = line.split('|')
        if len(fields) >= 2:
            states[fields[0].strip()] = fields[1].strip().split()[0].rstrip('+')
    if str(job_id) not in states:
        raise RuntimeError(f'Slurm JobId {job_id} has no verified status; recovery pending')
    relevant = [state for key, state in states.items()
                if key == str(job_id) or key.startswith(str(job_id) + '.')]
    return {'active': any(state not in TERMINAL for state in relevant),
            'state': states[str(job_id)], 'job_id': str(job_id),
            'successful': all(state == 'COMPLETED' for state in relevant)}


if __name__ == '__main__':
    if sys.argv[1] == 'submit':
        result = submit(Path(sys.argv[2]))
    else:
        result = status(sys.argv[2])
    print(json.dumps(result), flush=True)
