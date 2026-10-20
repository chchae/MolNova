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
    array = record.get('array_count') is not None
    ids = {line.strip() for line in query(
        ['squeue', '--noheader', '--name', name,
         '--format=%F' if array else '--format=%A']).splitlines() if line.strip()}
    since = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(record['started_at'] - 86400))
    rows = query(['sacct', *(['--array'] if array else []), '--noheader', '--parsable2', '--starttime', since,
                  '--name', name, '--format=JobID%64,JobName%128' if array else
                  '--format=JobIDRaw,JobName%128'])
    for line in rows.splitlines():
        fields = line.split('|')
        if len(fields) >= 2 and fields[1].strip() == name:
            identity = fields[0].strip()
            if array:
                match = re.fullmatch(r'(\d+)(?:_\d+)?', identity)
                if match:
                    ids.add(match[1])
                elif '.' not in identity:
                    raise RuntimeError('Unrecognized Slurm array history; automatic resubmission blocked')
            elif identity.isdigit():
                ids.add(identity)
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
        count = options.get('array_count')
        if count is not None and (isinstance(count, bool) or not isinstance(count, int)
                                  or count < 1 or count != len(config['entries'])):
            raise ValueError('Slurm Amber array count differs from saved entries')
        maximum = 384 if count is not None else 192
        if isinstance(cpus, bool) or not isinstance(cpus, int) or not 1 <= cpus <= maximum:
            raise ValueError(f'Slurm Amber requires 1..{maximum} CPUs')
        record = {'job_name': options['job_name'], 'started_at': time.time(),
                  'launching': True, 'job_id': None}
        if count is not None:
            record['array_count'] = count
        save(path, record)
        command = ['sbatch', '--parsable', '--job-name', record['job_name'],
                   '--partition', options['partition'], '--nodes=1', '--ntasks=1',
                   '--cpus-per-task', '1' if count is not None else str(cpus),
                   '--hint=nomultithread', '--mem=2G' if count is not None else '--mem=0',
                   '--chdir', str(root), '--output', str(root / ('slurm-%A_%a.out' if count is not None else 'slurm-%j.out')),
                   '--error', str(root / ('slurm-%A_%a.err' if count is not None else 'slurm-%j.err'))]
        if count is not None:
            command += ['--array', f'0-{count - 1}%{min(cpus, count)}']
        # Only new manifests pin a limit; saved submissions keep their protocol.
        if 'time_limit' in options:
            command.append('--time=' + options['time_limit'])
        command.append(str(root / 'amber.sbatch'))
        # Preserve the journal/uncertainty even on timeout or interrupted SSH.
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE,
                                         timeout=90, pass_fds=(lock.fileno(),))
        if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?\s*', output):
            raise RuntimeError('sbatch returned no unambiguous JobId; retained for reconciliation')
        record.update(job_id=output.strip().split(';')[0], launching=False)
        save(path, record)
        return record


def status(job_id, array_count=None):
    if not str(job_id).isdigit():
        raise ValueError('Invalid Slurm JobId')
    try:
        queued = query(['squeue', '--noheader', '--jobs', str(job_id), '--format=%T']).strip()
    except subprocess.CalledProcessError as exc:
        # Slurm may evict finished jobs from squeue before we poll them.
        # Only this specific absence permits an accounting lookup; transport,
        # permission and scheduler errors must remain uncertain.
        if 'Invalid job id specified' not in (exc.stderr or ''):
            raise
        queued = ''
    if queued:
        return {'active': True, 'state': queued, 'job_id': str(job_id)}
    if array_count is not None:
        return array_status(str(job_id), array_count)
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


def array_status(job_id, count):
    """Require every saved task and every reported step; arrays have no parent row."""
    if count < 1:
        raise ValueError('Invalid Slurm array count')
    rows = query(['sacct', '--array', '--noheader', '--parsable2', '--jobs', job_id,
                  '--format=JobID%64,State'])
    states = {}
    for line in rows.splitlines():
        fields = line.split('|')
        if len(fields) >= 2 and re.fullmatch(re.escape(job_id) + r'_\d+(?:\.[\w]+)?', fields[0].strip()):
            states[fields[0].strip()] = fields[1].strip().split()[0].rstrip('+')
    expected = {f'{job_id}_{index}' for index in range(count)}
    parents = {key for key in states if '.' not in key}
    if parents != expected or any(key.split('.')[0] not in expected for key in states):
        raise RuntimeError(f'Slurm array {job_id} has missing/unexpected task history; recovery pending')
    return {'active': any(state not in TERMINAL for state in states.values()),
            'state': ','.join(sorted({states[key] for key in expected})), 'job_id': job_id,
            'task_states': {key: states[key] for key in sorted(expected)},
            'task_successful': {key: all(value == 'COMPLETED' for name, value in states.items()
                                         if name == key or name.startswith(key + '.'))
                                for key in sorted(expected)},
            'successful': all(state == 'COMPLETED' for state in states.values())}


if __name__ == '__main__':
    if sys.argv[1] == 'submit':
        result = submit(Path(sys.argv[2]))
    else:
        result = status(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(json.dumps(result), flush=True)
