"""Stdlib-only Slurm submission/reconciliation on the submission host."""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amber_slurm_control import recover, save, status


def submit(root):
    with (root/'submit.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'busy': True}
        path = root/'submission.json'
        request = json.loads((root/'request.json').read_text())
        record = json.loads(path.read_text()) if path.exists() else {}
        if record.get('job_id'):
            return record
        if record.get('launching'):
            job_id = recover(record)
            if job_id:
                record.update(job_id=job_id, launching=False)
                save(path, record)
                return record
            if time.time()-record['started_at'] < 60:
                return {'busy': True}
            save(root/f'unsubmitted-{time.time_ns()}.json', record)
        record = dict(job_name=request['job_name'], started_at=time.time(), launching=True, job_id=None)
        save(path, record)
        hosts = request['hosts']
        slots = list(hosts.values())
        if (not hosts or len(hosts)>2 or len(set(slots))!=1
                or any(isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=192 for n in slots)):
            raise ValueError('Slurm requires one or two equally sized pools of 1..192 CPUs')
        command = ['sbatch', '--parsable', '--no-requeue', '--job-name', request['job_name'],
                   '--partition', request['partition'], '--nodes', str(len(hosts)),
                   '--nodelist', ','.join(hosts), '--ntasks-per-node=1',
                   '--cpus-per-task', str(slots[0]), '--hint=nomultithread',
                   '--mem', str(request['memory_mb']), '--time', request['time_limit'],
                   '--chdir', str(root), '--output', str(root/'job.out'),
                   '--error', str(root/'job.err'), str(root/'run.sbatch')]
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE,
                                         timeout=90, pass_fds=(lock.fileno(),))
        if not re.fullmatch(r'\d+(?:;[A-Za-z0-9_.-]+)?\s*', output):
            raise RuntimeError('No unambiguous Slurm JobId; submission retained for reconciliation')
        record.update(job_id=output.strip().split(';')[0], launching=False)
        save(path, record)
        return record


if __name__ == '__main__':
    action, value = sys.argv[1:]
    if action=='cancel':
        if not value.isdigit():raise ValueError('Invalid Slurm JobId')
        subprocess.run(['scancel',value],check=True,timeout=30)
        result = {'cancel_requested':True,'job_id':value}
    elif action=='submit':result = submit(Path(value))
    elif action=='status':result = status(value)
    else:raise ValueError('Unknown Slurm control action')
    print(json.dumps(result))
