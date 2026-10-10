"""Durable LigPrep submission identity for crash/restart recovery."""
import json
import re
import subprocess
from pathlib import Path


def job_file(directory):
    return Path(directory) / 'ligprep_job.json'


def load_job(directory):
    path = job_file(directory)
    return json.loads(path.read_text()) if path.exists() else None


def _save(directory, record):
    path = job_file(directory)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record))
    temporary.replace(path)


def run_or_recover(command, directory, compound_ids, output):
    """Return False while a saved job is live; never repeat an uncertain launch."""
    directory, output = Path(directory), Path(output)
    record = load_job(directory)
    if record is not None:
        if not record.get('job_id'):
            raise RuntimeError('Interrupted LigPrep launch without saved JobId; '
                               'reconcile ligprep_job.json with JobServer before resubmitting.')
        jsc = Path(record['command'][0]).parent / 'jsc'
        result = subprocess.run([str(jsc), 'info', '--json', record['job_id']],
                                cwd=directory, check=True, capture_output=True, text=True)
        jobs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        status = next(job['status'] for job in jobs if job['jobId'] == record['job_id'])
        if status not in {'DONE', 'FAILED', 'CANCELED', 'STOPPED'}:
            print(f"LigPrep job {record['job_id']}: {status}; checking on next worker run.")
            return False
        subprocess.run([str(jsc), 'download', '--cwd', record['job_id']],
                       cwd=directory, check=True, capture_output=True, text=True)
        if status != 'DONE':
            job_file(directory).replace(Path(directory) / 'ligprep_job.failed.json')
            for log in directory.glob('*.log'):
                log.rename(log.with_suffix('.log.failed'))
            raise RuntimeError(f"LigPrep job {record['job_id']} ended with {status}")
    else:
        logs = list(directory.glob('*.log'))
        if logs:
            # Legacy -WAIT jobs may still be alive after a worker crash.
            for log in logs:
                match = re.search(r'^JobId\s*:\s*(\S+)', log.read_text(errors='replace'),
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    _save(directory, {'command': [str(value) for value in command],
                                      'compound_ids': list(compound_ids), 'job_id': match.group(1)})
                    return run_or_recover(command, directory, compound_ids, output)
            raise RuntimeError('Legacy LigPrep logs have no JobId; reconcile existing job before resubmitting.')
        command = [str(value) for value in command]
        record = {'command': command, 'compound_ids': list(compound_ids), 'job_id': None}
        _save(directory, record)  # Write before launching, including uncertain launches.
        output.unlink(missing_ok=True)
        print('$', ' '.join(command), flush=True)
        with subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                match = re.search(r'^JobId:\s*(\S+)', line, re.IGNORECASE)
                if match:
                    record['job_id'] = match.group(1)
                    _save(directory, record)
            code = proc.wait()
        if code:
            # The external job may survive a failed local -WAIT client.
            if record['job_id']:
                return False
            raise RuntimeError('LigPrep failed without a saved JobId; reconcile submission before retrying.')
    if not output.is_file() or not output.stat().st_size:
        job_file(directory).unlink()
        raise RuntimeError('Completed LigPrep job produced no output.')
    return True  # Retain record until the stage commits its compound states.
