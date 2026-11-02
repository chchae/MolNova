"""Explicit retry of terminal Amber failures; native work has its own immutable run.

The normal pipeline and already submitted Prime selections are not rescheduled.
This worker only imports scores for the explicitly retried failures.
"""
import argparse
import json
from pathlib import Path
import time

from molnova import _core, amber_gbsa, amber_slurm, database
from molnova.paths import stage_directory


def poll(args, iteration, source):
    parent = stage_directory(args.output, iteration, 'amber_gbsa')
    directory = parent / 'timeout-retry'
    directory.mkdir(parents=True, exist_ok=True)
    original = json.loads(source.read_text())
    if (original.get('finished') is not True or original['iteration'] != iteration
            or Path(original['db_path']).resolve() != args.db_path.resolve()
            or original.get('mode') != amber_slurm.ARRAY):
        raise ValueError('Retry requires this project/iteration finished Amber array manifest')
    amber_gbsa.verify_sources(args, iteration, original)
    if not args.use_slurm:
        raise ValueError('Checkpoint retry requires a Slurm compute allocation')
    record = amber_gbsa.load_job(directory)
    completed = list(directory.glob('run-*.completed.json'))
    if record is None and completed:
        print('Amber retry already collected; no resubmission.', flush=True)
        return True
    if record is None:
        native = amber_slurm.job_state(parent, original)
        remote = original['remote_dir']
        import shlex
        response = amber_gbsa.ssh(
            original['settings'], f'cd {shlex.quote(remote)} && python3 - collect {shlex.quote(remote)}',
            input=Path(amber_slurm.__file__).with_name('amber_array.py').read_text(),
            capture_output=True, text=True)
        if native['active'] or json.loads(response.stdout)['active']:
            raise RuntimeError('Original Amber jobs or inherited locks are still active; retry withheld')
        ids = database.failed_amber_retry_ids(args.db_path, iteration, original['compound_ids'])
        if not ids:
            print('No unscored terminal Amber failures to retry.', flush=True)
            return True
        record = amber_slurm.new_record(args, iteration, directory, ids)
    if not record.get('prepared'):
        # These settings are frozen before any external scientific work starts.
        record.update(retry_failed=True, retry_of=original['run_id'],
                      minimization_timeout_s=600, slurm_time_limit='01:00:00')
        amber_gbsa.save(directory / 'amber_job.json', record)
    if (not record.get('retry_failed') or record.get('retry_of') != original['run_id']
            or record.get('minimization_timeout_s') != 600):
        raise RuntimeError('Saved retry policy/ownership differs; preserve saved work')
    done = amber_gbsa.poll(args, iteration, directory, record)
    snapshot = json.loads((directory / 'remote_snapshot.json').read_text())
    scores = [r for r in snapshot['results'] if 'score' in r]
    truncated = sum(r.get('minimization', {}).get('status') == 'time_limit_checkpoint' for r in scores)
    print(f'Iteration {iteration}: Amber retry scores={len(scores)}/{len(record["compound_ids"])}, '
          f'checkpoint scores={truncated}, collected={done}.', flush=True)
    if done:
        amber_gbsa.write_rankings(args, iteration, parent)
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project_toml', type=Path)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    ns = parser.parse_args(argv)
    args = _core.configure_project(ns.project_toml, None)
    while True:
        try:
            with database.stage_lock(args.db_path, 'amber_gbsa'):
                done = poll(args, ns.iteration, ns.source.resolve())
        except Exception as exc:
            if not ns.watch:
                raise
            # Retain the exact native identity through transient transport,
            # accounting or collection failures. Never create a replacement job.
            print(f'Amber retry collection pending: {type(exc).__name__}: {exc}', flush=True)
            time.sleep(30)
            continue
        if done or not ns.watch:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
