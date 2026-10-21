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
    if ('prime_gbsa' in name or 'mmgbsa' in name or 'prime_mmgbsa' in command
            or 'prime_gbsa' in program or 'mmgbsa' in program):
        return 'prime_gbsa'
    if name.startswith('glide') or '/glide ' in command or 'glide_wrap.py' in command or program == 'glide':
        return 'glide'
    return None


def _local_active_jobs(suite):
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



def _md_registry():
    return Path.home() / '.cache' / 'molnova' / 'md-remotes.json'


def register_md_remote(settings):
    # Same submission mutex as Glide/Prime. Registrations remain durable so a
    # controller crash cannot hide active or uncertain Slurm jobs.
    path = Path(tempfile.gettempdir()) / f'molnova-schrodinger-submit-{os.getuid()}.lock'
    with path.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        registry = _md_registry()
        registry.parent.mkdir(parents=True,exist_ok=True)
        saved = json.loads(registry.read_text()) if registry.exists() else {}
        saved[settings['ssh_host']+'|'+settings['suite']+'|'+settings['root']] = {
            key:settings[key] for key in ('ssh_host','suite','helper','root')}
        from molnova.md_gbsa_native import save
        save(registry,saved)



def unregister_md_remote(settings):
    path = Path(tempfile.gettempdir()) / f'molnova-schrodinger-submit-{os.getuid()}.lock'
    with path.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        registry = _md_registry()
        if registry.exists():
            saved=json.loads(registry.read_text())
            saved.pop(settings['ssh_host']+'|'+settings['suite']+'|'+settings['root'],None)
            from molnova.md_gbsa_native import save
            save(registry,saved)

def active_jobs(suite):
    rows = _local_active_jobs(suite)
    registry = _md_registry()
    if registry.exists():
        from molnova.md_gbsa import remote_call, response
        for settings in json.loads(registry.read_text()).values():
            for job in response(remote_call(settings,'jobs',settings['root'])):
                rows.append(dict(job,molnova_md_remote=True))
    return rows

def belongs_to_directory(job, directory):
    launch = job.get('spec', {}).get('launchParams', {}).get('launchDirectory')
    if launch:
        return Path(launch).resolve() == Path(directory).resolve()
    return str(Path(directory).resolve()) + '/' in job.get('commandLine', '')


@contextmanager
def submission_guard(suite, stage, *, md_controller=False):
    # Shared across projects for this user, not held while calculations run.
    path = Path(tempfile.gettempdir()) / f'molnova-schrodinger-submit-{os.getuid()}.lock'
    with path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SubmissionBlocked('Another Schrödinger submission check is in progress') from exc
        try:
            opposite = 'glide' if stage == 'prime_gbsa' else 'prime_gbsa'
            try:
                jobs = active_jobs(suite)
            except Exception as exc:
                raise SubmissionBlocked(f'Cannot verify external jobs; submissions withheld: {exc}') from exc
            live = [j for j in jobs if job_kind(j) == opposite or
                    (stage == 'prime_gbsa' and not md_controller and
                     j.get('molnova_md_remote') and job_kind(j) == 'prime_gbsa')]
            if live:
                raise SubmissionBlocked(f"{job_kind(live[0])} still active in JobServer: " +
                                        ', '.join(j['jobId'] for j in live))
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
