"""Clear orphaned iteration files before the first generation, preserving recovery."""
import json
from pathlib import Path
import re
import time

from molnova import database
from molnova import schrodinger_guard as guard


def prepare_iteration_directory(args, iteration):
    """Run under the generation stage lock, after preceding-iteration gates."""
    if iteration < 1:
        raise ValueError('Iteration workspace cleanup requires iteration >= 1')
    if database.iteration_has_saved_work(args.db_path, iteration):
        return None  # Resume exactly the saved iteration, including zero-row plans.
    root = Path(args.output) / f'iter{iteration}'
    if root.is_symlink():
        raise RuntimeError(f'Iteration directory is a symlink; files retained: {root}')
    if not root.exists() or not any(root.iterdir()):
        database.mark_iteration_workspace_prepared(args.db_path, iteration)
        return None

    # Configuration inputs must never be relocated while clearing working output.
    for key in ('db_path', 'toml_file', 'grid', 'reference_poses', 'libinvent_prior',
                'sidechain_library', 'mmgbsa_receptor', 'amber_receptor', 'aizynth_config'):
        source = getattr(args, key, None)
        if source is not None and Path(source).resolve().is_relative_to(root.resolve()):
            raise RuntimeError(f'Configured {key} is inside {root}; move the input before cleanup')

    paths = list(root.rglob('*'))
    files = [path for path in paths if path.is_file()]
    jobs = {}
    unidentified_logs = []
    for path in files:
        if path.name == 'amber_job.json' or path.name.endswith('.remote-dir.txt'):
            raise RuntimeError(f'Remote work requires reconciliation; files retained: {path}')
        if path.name in ('ligprep_job.json', 'glide_job.json', 'prime_job.json',
                         'ligprep_job.tmp', 'glide_job.tmp', 'prime_job.tmp'):
            record = json.loads(path.read_text())
            if not record.get('job_id'):
                raise RuntimeError(f'Saved or uncertain submission requires recovery; files retained: {path}')
            jobs[record['job_id']] = path.parent
        if path.suffix == '.log':
            ids = re.findall(r'^JobId\s*:\s*(\S+)', path.read_text(errors='replace'),
                             re.IGNORECASE | re.MULTILINE)
            jobs.update((job_id, path.parent) for job_id in ids)
            if not ids and (path.name.startswith('ligprep') or path.name.startswith('glide_constrained')
                            or path.name.startswith('mmgbsa_input') or path.name == 'glide_launch.log'):
                unidentified_logs.append(path)
    for path in unidentified_logs:
        if path.parent not in jobs.values():
            raise RuntimeError(f'Calculation log has no recoverable JobId; files retained: {path}')

    # Check all native jobs before moving a single file, including children of
    # terminal parents. A lookup failure propagates without changing anything.
    directories = {root, *(path for path in paths if path.is_dir())}
    live = guard.active_jobs(args.schrodinger)
    if any(guard.belongs_to_directory(job, folder) for job in live for folder in directories):
        raise RuntimeError(f'Iteration has live external jobs; files retained: {root}')
    for job_id, folder in jobs.items():
        details = guard.job_details(args.schrodinger, job_id, folder)
        if (not any(job.get('jobId') == job_id for job in details)
                or any(job.get('status') not in guard.TERMINAL for job in details)):
            raise RuntimeError(f'Iteration job {job_id} is live or uncertain; files retained: {root}')

    archive_root = Path(args.output) / '.previous'
    if archive_root.is_symlink():
        raise RuntimeError(f'Iteration archive is a symlink; files retained: {archive_root}')
    archive_root.mkdir(exist_ok=True)
    archive = archive_root / f'iter{iteration}-{time.time_ns()}'
    root.rename(archive)
    root.mkdir()
    database.mark_iteration_workspace_prepared(args.db_path, iteration)
    print(f'Iteration {iteration}: cleared previous calculation files before generation; '
          f'preserved in {archive}.', flush=True)
    return archive
