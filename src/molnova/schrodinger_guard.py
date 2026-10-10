"""Serialize submissions and fence live Glide/Prime jobs using JobServer."""
import fcntl
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

TERMINAL = {'DONE', 'FAILED', 'CANCELED', 'STOPPED'}


class SubmissionBlocked(RuntimeError):
    pass


def job_details(suite, job_id, directory=None):
    result = subprocess.run([str(Path(suite) / 'jsc'), 'info', '--json', job_id],
                            cwd=directory, check=True, capture_output=True, text=True, timeout=30)
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if not any(row.get('jobId') == job_id for row in rows):
        raise RuntimeError(f'JobServer returned no parent {job_id}')
    return rows


def job_kind(job):
    name = job.get('jobName', '').lower()
    command = job.get('commandLine', '').lower()
    program = job.get('spec', {}).get('taskSpec', {}).get('programName', '').lower()
    if 'mmgbsa' in name or 'prime_mmgbsa' in command or 'mmgbsa' in program:
        return 'mmgbsa'
    if name.startswith('glide') or '/glide ' in command or 'glide_wrap.py' in command or program == 'glide':
        return 'glide'
    return None


def active_jobs(suite):
    result = subprocess.run([str(Path(suite) / 'jsc'), 'list', '--id-only'],
                            capture_output=True, text=True, timeout=30)
    if getattr(result, 'returncode', 0):
        message = (result.stdout + result.stderr).strip()
        if result.returncode == 1 and message == 'No jobs matching your search criteria were found.':
            return []
        result.check_returncode()
    rows = {}
    inspected = set()
    for job_id in result.stdout.splitlines():
        job_id = job_id.strip()
        if not job_id or job_id in inspected:
            continue
        for job in job_details(suite, job_id):
            inspected.add(job['jobId'])
            if job.get('status') not in TERMINAL:
                rows[job['jobId']] = job
    return list(rows.values())


def belongs_to_directory(job, directory):
    launch = job.get('spec', {}).get('launchParams', {}).get('launchDirectory')
    if launch:
        return Path(launch).resolve() == Path(directory).resolve()
    return str(Path(directory).resolve()) + '/' in job.get('commandLine', '')


@contextmanager
def submission_guard(suite, stage):
    # Shared across projects for this user, not held while calculations run.
    path = Path(tempfile.gettempdir()) / f'molnova-schrodinger-submit-{os.getuid()}.lock'
    with path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SubmissionBlocked('Another Schrödinger submission check is in progress') from exc
        try:
            opposite = 'glide' if stage == 'mmgbsa' else 'mmgbsa'
            try:
                jobs = active_jobs(suite)
            except Exception as exc:
                raise SubmissionBlocked(f'Cannot verify external jobs; submissions withheld: {exc}') from exc
            live = [j for j in jobs if job_kind(j) == opposite]
            if live:
                raise SubmissionBlocked(f"{opposite} still active in JobServer: " +
                                        ', '.join(j['jobId'] for j in live))
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
