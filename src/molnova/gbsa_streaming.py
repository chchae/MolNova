"""Import immutable completed Prime subjob outputs while the parent runs."""
import re
import subprocess
from pathlib import Path

from molnova import _core
from molnova.prime_async import _save
from molnova.stages._logging import job_elapsed, format_elapsed


def _fetch_output(jsc, directory, record, child, output):
    """The parent may have already collected the child's output; try both."""
    filename = child['jobName'] + '-out.maegz'
    temporary = output.with_name('result.pending.maegz')
    for job_id in (record['job_id'], child['jobId']):
        with temporary.open('wb') as stream:
            result = subprocess.run(
                [str(jsc), 'tail-file', '--name', filename, '--force', job_id],
                cwd=directory, stdout=stream, stderr=subprocess.PIPE, timeout=30,
            )
        if result.returncode == 0 and temporary.stat().st_size:
            temporary.replace(output)
            return True
    temporary.unlink(missing_ok=True)
    return False


def collect_completed_subjobs(args, iteration, directory, record, jobs):
    """Commit scores before recording imported children, making restarts safe."""
    parent = next(job for job in jobs if job['jobId'] == record['job_id'])
    if parent['status'] == 'DONE':
        return  # The final parent output is authoritative and now available.
    directory = Path(directory)
    jsc = Path(record['command'][0]).parent / 'jsc'
    processed = set(record.get('imported_subjobs', []))
    input_stem = Path(record['command'][1]).stem
    total_updated = 0
    for child in jobs:
        if (child.get('parentJobId') != record['job_id'] or child.get('status') != 'DONE'
                or child['jobId'] in processed
                or not re.fullmatch(re.escape(input_stem) + r'\.\d+', child.get('jobName', ''))):
            continue
        # JobServer IDs and filenames must never escape this submission folder.
        if not re.fullmatch(r'[A-Za-z0-9_-]+', child['jobId']):
            raise ValueError('Invalid Prime subjob ID')
        folder = directory / 'subjobs' / record['job_id'] / child['jobId']
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / 'result.maegz'
        if not output.exists() and not _fetch_output(jsc, directory, record, child, output):
            print(f"Prime subjob {child['jobId']}: output not yet available; retrying on next poll.")
            continue
        extractor = _core.create_mmgbsa_score_extractor(folder)
        scores = folder / 'scores.tsv'
        _core.run_command([args.schrodinger / 'run', extractor, output, scores], cwd=folder)
        updated = _core.update_gbsa_scores(scores, args.db_path, allowed_ids=record['compound_ids'])
        # Commit succeeded. Repeating this import after a crash preserves scores.
        processed.add(child['jobId'])
        record['imported_subjobs'] = sorted(processed)
        _save(directory, record)
        total_updated += updated
        elapsed = job_elapsed(child)
        timing = f'; calculation elapsed={format_elapsed(elapsed)}' if elapsed is not None else ''
        print(f"Iteration {iteration}: Prime subjob {child['jobName']} completed; "
              f"GBSA DB scores updated={updated}{timing}.", flush=True)
    return total_updated
