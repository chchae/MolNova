"""Clear orphaned iteration files before the first generation, preserving recovery."""
import json
from pathlib import Path
import re
import time
import csv
from contextlib import ExitStack

from molnova.paths import iteration_directory, stage_in_iteration
from molnova import database
from molnova import schrodinger_guard as guard


def prepare_iteration_directory(args, iteration):
    """Run under the generation stage lock, after preceding-iteration gates."""
    if iteration < 1:
        raise ValueError('Iteration workspace cleanup requires iteration >= 1')
    deleted = database.iteration_was_deleted(args.db_path, iteration)
    if database.iteration_has_saved_work(args.db_path, iteration) and not deleted:
        return None  # Resume exactly the saved iteration, including zero-row plans.
    with ExitStack() as locks:
        if deleted:
            for stage in ('druglike', 'synthetic_feasibility', 'ligprep', 'glide', 'amber_gbsa', 'prime_gbsa', 'fep'):
                locks.enter_context(database.stage_lock(args.db_path, stage))
        return _prepare(args, iteration, deleted)


def _prepare(args, iteration, deleted):
    root = iteration_directory(Path(args.output), iteration)
    if root.is_symlink():
        raise RuntimeError(f'Iteration directory is a symlink; files retained: {root}')
    if not deleted and (not root.exists() or not any(root.iterdir())):
        database.mark_iteration_workspace_prepared(args.db_path, iteration)
        return None

    # Configuration inputs must never be relocated while clearing working output.
    for key in ('db_path', 'toml_file', 'grid', 'reference_poses', 'libinvent_prior',
                'sidechain_library', 'prime_gbsa_receptor', 'amber_receptor', 'aizynth_config'):
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
        if path.name in ('ligprep_job.json', 'glide_job.json', 'prime_gbsa_job.json',
                         'ligprep_job.tmp', 'glide_job.tmp', 'prime_gbsa_job.tmp'):
            record = json.loads(path.read_text())
            if not record.get('job_id'):
                raise RuntimeError(f'Saved or uncertain submission requires recovery; files retained: {path}')
            jobs[record['job_id']] = path.parent
        if path.suffix == '.log':
            ids = re.findall(r'^JobId\s*:\s*(\S+)', path.read_text(errors='replace'),
                             re.IGNORECASE | re.MULTILINE)
            jobs.update((job_id, path.parent) for job_id in ids)
            if not ids and (path.name.startswith('ligprep') or path.name.startswith('glide')
                            or path.name.startswith('prime_gbsa_') or path.name == 'glide_launch.log'):
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
    for job in live:
        launch = job.get('spec', {}).get('launchParams', {}).get('launchDirectory')
        if launch and Path(launch).resolve().is_relative_to(root.resolve()):
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
    archive = archive_root / f'{root.name}-{time.time_ns()}'
    report_ids = []
    report = stage_in_iteration(root, 'druglike') / 'druglike_selection.csv'
    if deleted and report.exists():
        with report.open(newline='') as stream:
            report_ids = [int(row['id']) for row in csv.DictReader(stream)]
    had_root = root.exists()
    if had_root:
        root.rename(archive)
    else:
        archive.mkdir()
    root.mkdir(parents=True)
    try:
        if deleted:
            database.reset_deleted_iteration(args.db_path, iteration,
                                             archive / f'database-before-reset-{time.time_ns()}.sqlite', report_ids)
        else:
            database.mark_iteration_workspace_prepared(args.db_path, iteration)
    except BaseException:
        # No worker can modify this iteration while the stage locks are held.
        # Restore its original files if the DB transaction/backup did not finish.
        root.rmdir()
        if had_root:
            archive.rename(root)
        raise
    print(f'Iteration {iteration}: cleared previous calculation files before generation; '
          f'preserved in {archive}.', flush=True)
    if deleted:
        print(f'Iteration {iteration}: backed up SQLite/WAL and reset orphaned selection, '
              'scaffold and generation progress; starting a fresh population.', flush=True)
    return archive
