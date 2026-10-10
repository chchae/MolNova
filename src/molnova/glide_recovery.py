"""Track native Glide identity; cached output never proves a live job finished."""
import json
import re
import subprocess
import time
from pathlib import Path

from molnova.schrodinger_guard import (active_jobs, belongs_to_directory,
    job_details, job_kind, submission_guard, TERMINAL)


def record_file(directory):
    return Path(directory) / 'glide_job.json'


def load_record(directory):
    path = record_file(directory)
    return json.loads(path.read_text()) if path.exists() else None


def save_record(directory, record):
    path = record_file(directory)
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(record))
    pending.replace(path)


def group_status(suite, directory, log_file, output_file, live=None):
    """Return a native status, UNKNOWN, or None for an unsubmitted group."""
    directory = Path(directory)
    live = active_jobs(suite) if live is None else live
    matches = [j for j in live if job_kind(j) == 'glide' and
               belongs_to_directory(j, directory)]
    record = load_record(directory)
    if matches:
        parents = [j for j in matches if not j.get('parentJobId')]
        if parents:
            # Recover a legacy/duplicate launch even if the local log is older.
            parent = max(parents, key=lambda j: j.get('timeCreated', ''))
            if record is None or record.get('job_id') and record['job_id'] != parent['jobId']:
                record = {'job_id': parent['jobId'], 'recovered': True, 'downloaded': False}
                save_record(directory, record)
        return 'RUNNING'
    if record is None and log_file.exists():
        match = re.search(r'^JobId\s*:\s*(\S+)', log_file.read_text(errors='replace'), re.MULTILINE)
        if match:
            record = {'job_id': match.group(1), 'recovered': True, 'downloaded': False}
            save_record(directory, record)
    if record is not None:
        if not record.get('job_id'):
            return 'UNKNOWN'  # Never repeat an uncertain launch.
        jobs = job_details(suite, record['job_id'], directory)
        if any(j.get('status') not in TERMINAL for j in jobs):
            return 'RUNNING'
        parent = next(j for j in jobs if j['jobId'] == record['job_id'])
        if parent['status'] == 'DONE' and not record.get('downloaded'):
            subprocess.run([str(Path(suite) / 'jsc'), 'download', '--cwd', record['job_id']],
                           cwd=directory, check=True, capture_output=True, text=True, timeout=60)
            record['downloaded'] = True
            save_record(directory, record)
        return parent['status']
    # Legacy files without JobId are usable only after the global live-job check.
    if output_file.exists() and output_file.stat().st_size:
        return 'DONE'
    if log_file.exists():
        text = log_file.read_text(errors='replace')
        if 'Finished at:' in text:
            return 'DONE'
        return 'UNKNOWN'
    return None


def submit(suite, command, directory):
    directory = Path(directory)
    with submission_guard(suite, 'glide'):
        # Recheck under the same submission mutex as MM-GBSA.
        live = active_jobs(suite)
        if any(job_kind(j) == 'glide' and belongs_to_directory(j, directory) for j in live):
            raise RuntimeError('Glide group already has a live external job; recover it instead')
        existing = load_record(directory)
        if existing and not existing.get('job_id'):
            raise RuntimeError('Glide launch has no saved JobId; reconcile before resubmission')
        if existing:
            if any(j.get('status') not in TERMINAL for j in job_details(suite, existing['job_id'], directory)):
                raise RuntimeError('Glide group saved job still running')
        archive = directory / f'previous-{time.time_ns()}'
        old = list(directory.glob('*_lib.maegz')) + list(directory.glob('glide_constrained*.log'))
        if old:
            archive.mkdir()
            for path in old:
                path.replace(archive / path.name)
        record = {'command': [str(x) for x in command], 'job_id': None,
                  'submitted_at': time.time(), 'downloaded': False}
        save_record(directory, record)
        result = subprocess.run(record['command'], cwd=directory, check=True,
                                capture_output=True, text=True)
        print(result.stdout, end='', flush=True)
        if result.stderr:
            print(result.stderr, end='', flush=True)
        match = re.search(r'^JobId:\s*(\S+)', result.stdout, re.MULTILINE | re.IGNORECASE)
        if not match:
            raise RuntimeError('Glide submission returned no JobId; reconcile before retrying')
        record['job_id'] = match.group(1)
        save_record(directory, record)
        return record['job_id']
