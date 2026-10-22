"""Independent Amber screening stage; Prime keeps its existing PRIME_GBSA worker."""
import argparse
from pathlib import Path

from molnova.paths import stage_directory, stage_manifests
from molnova import _core as c, amber_gbsa, database
from molnova.prime_gbsa_async import load_job as load_prime
from molnova.stages._logging import timed_stage, work_started


@timed_stage('Amber GBSA worker')
def _run_stage(args, iteration):
    directory = stage_directory(args.output, iteration, 'amber_gbsa')
    saved = amber_gbsa.load_job(directory)
    if not args.amber_gbsa_enabled and saved is None:
        print('Amber GBSA disabled; set amber_gbsa=true in the project TOML.')
        return
    if iteration < 0:
        raise ValueError('Amber GBSA requires iteration >= 0')
    directory.mkdir(parents=True, exist_ok=True)
    reason = database.docking_completion_reason(args.db_path, iteration, args.target_count) if iteration > 0 else None
    if reason or any(database.stage_is_running(args.db_path, stage) for stage in ('generate', 'synthetic_feasibility', 'ligprep', 'glide', 'prime_gbsa')):
        print(f'Amber GBSA waiting: {reason or "upstream/Prime worker still running"}.')
        return
    from molnova import schrodinger_guard as guard
    from molnova.glide_recovery import load_record
    for path in (stage_directory(args.output, iteration, 'glide')).glob('*/glide_job.json'):
        record = load_record(path.parent)
        if record.get('downloaded'):
            continue
        if not record.get('job_id'):
            print('Amber GBSA waiting: saved Glide submission identity uncertain.')
            return
        details = guard.job_details(args.schrodinger, record['job_id'])
        if not details or any(job.get('status') not in guard.TERMINAL for job in details):
            print(f'Amber GBSA waiting: saved Glide parent/children still active: {record["job_id"]}')
            return
    if load_prime(stage_directory(args.output, iteration, 'prime_gbsa')) is not None or database.has_running_prime_gbsa(args.db_path, iteration):
        print('Amber GBSA waiting for existing Prime claims/submissions. Recover them with molnova prime_gbsa first.')
        return
    record = saved
    if record is None:
        import json
        for archived in directory.glob('run-*.completed.json'):
            amber_gbsa.verify_sources(args, iteration, json.loads(archived.read_text()))
        other_records = [p for p in stage_manifests(args.output, 'amber_gbsa', 'amber_job.json') if p.parent != directory]
        if other_records:
            print(f'Amber GBSA waiting for another saved iteration: {other_records[0]}')
            return
        database.reset_unsubmitted_amber(args.db_path, iteration)
        if iteration == 0:
            amber_gbsa.prepare_reference_receptor(args)
        if not args.amber_receptor.is_file():
            raise FileNotFoundError(f'Amber receptor not found: {args.amber_receptor}')
        ids = database.claim_amber_candidates(args.db_path, iteration, args.target_count, args.amber_keep_count)
        if ids:
            record = amber_gbsa.new_record(args, iteration, directory, ids)
    if record is not None:
        work_started(f'Iteration {iteration}: Amber screening/recovery of {len(record["compound_ids"])} compounds on {args.amber_remote["host"]}.')
        if not amber_gbsa.poll(args, iteration, directory, record):
            print('Amber GBSA remote work pending; persisted results will be recovered on the next poll.')
            return
    if not args.amber_gbsa_enabled:
        print('Amber GBSA disabled; saved submission recovered; no new candidates claimed.')
        return
    reason = database.amber_completion_reason(args.db_path, iteration, args.target_count, args.amber_keep_count)
    if reason:
        raise RuntimeError(reason)
    amber_gbsa.write_rankings(args, iteration, directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Remote AmberTools GBSA screening')
    parser.add_argument('project_toml', type=Path)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--schrodinger', type=Path)
    ns = parser.parse_args(argv)
    args = c.configure_project(ns.project_toml, ns.schrodinger)
    with database.stage_lock(args.db_path, 'amber_gbsa'):
        _run_stage(args, ns.iteration)


if __name__ == '__main__':
    main()
