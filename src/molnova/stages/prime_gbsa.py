#!/usr/bin/env python3
import argparse
from pathlib import Path
from molnova.paths import iteration_directory, stage_directory
from molnova import _core as c
from molnova import database
from molnova.stages._logging import work_started, timed_stage, progress
from molnova.states import CompoundState as State
from molnova.prime_gbsa_async import (PrimeAtomTypingError, PrimeReconciliationPending,
                                load_job, reconcile_saved_job)


def eligible_iteration(args):
    with c.open_sqlite(args.db_path) as conn:
        iterations = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT iteration FROM compound "
                "WHERE iteration=0 OR (iteration>0 AND docking_score IS NOT NULL) ORDER BY iteration"
            )
        ]
    for iteration in iterations:
        if (database.has_running_prime_gbsa(args.db_path, iteration)
                or database.prime_gbsa_candidates(args.db_path, iteration, args.prime_gbsa_input_count,
                                              getattr(args, "target_count", 1), database.amber_keep_count(args))):
            return iteration
    return None


@timed_stage("PRIME_GBSA worker")
def _run_stage(args, requested_iteration):
    if requested_iteration is None:
        with c.open_sqlite(args.db_path) as conn:
            iterations = [row[0] for row in conn.execute(
                "SELECT DISTINCT iteration FROM compound WHERE iteration=0 "
                "OR (iteration>0 AND docking_score IS NOT NULL) ORDER BY iteration"
            )]
        processed = False
        for iteration in iterations:
            if (load_job(stage_directory(args.output, iteration, 'prime_gbsa')) is not None
                    or database.has_running_prime_gbsa(args.db_path, iteration)
                    or database.prime_gbsa_candidates(args.db_path, iteration, args.prime_gbsa_input_count,
                                                  getattr(args, "target_count", 1), database.amber_keep_count(args))
                    or (iteration > 0 and database.docking_completion_reason(
                        args.db_path, iteration, getattr(args, "target_count", 1)) is not None)):
                processed = True
                _run_stage(args, iteration)
        if not processed:
            print("No completed docking iteration requires prime_gbsa.")
        return
    iteration = requested_iteration if requested_iteration is not None else eligible_iteration(args)
    if iteration is None:
        print("No completed docking iteration requires prime_gbsa.")
        return

    directory = stage_directory(args.output, iteration, 'prime_gbsa')
    enabled = getattr(args, 'prime_gbsa_enabled', True)
    if not enabled and load_job(directory) is None:
        if database.has_running_prime_gbsa(args.db_path, iteration):
            raise RuntimeError('Prime disabled but running claims have no saved submission; reconcile before advancing')
        print('Prime GBSA disabled; no saved submission to recover.')
        return
    target_count = getattr(args, "target_count", 1)
    reason = database.docking_completion_reason(args.db_path, iteration, target_count) if iteration > 0 else None
    if reason is not None:
        print(f"Iteration {iteration}: prime_gbsa waiting: {reason}.")
        return
    try:
        active = reconcile_saved_job(directory, args.db_path, iteration)
    except PrimeReconciliationPending as exc:
        print(str(exc))
        return
    if enabled:
        database.reset_unsubmitted_prime_gbsa(
            args.db_path, iteration, active["compound_ids"] if active else (),
        )
    if active is not None:
        # Recover the exact saved submission, including jobs from the former
        # streaming policy. Never overwrite an external job or erase its scores.
        pending = database.claim_compounds(
            args.db_path, active["compound_ids"],
            expected_state=(State.REFERENCE, State.DOCKED, State.PRIME_GBSA_RUNNING) if iteration == 0 else (State.DOCKED, State.PRIME_GBSA_RUNNING),
            claimed_state=State.PRIME_GBSA_RUNNING,
        )
    else:
        if not enabled:
            print('Prime GBSA disabled; saved submission retired; no new candidates claimed.')
            return
        # Disabling future Amber screening must not bypass saved/live work.
        from molnova.amber_gbsa import load_job as load_amber
        if (load_amber(stage_directory(args.output, iteration, 'amber_gbsa')) is not None
                or database.stage_is_running(args.db_path, 'amber_gbsa')):
            print(f'Iteration {iteration}: Prime waiting for Amber remote work to retire.')
            return
        pending = database.claim_prime_gbsa_candidates(
            args.db_path, iteration, args.prime_gbsa_input_count, target_count, database.amber_keep_count(args),
        )
    if not pending and active is not None:
        # A crash may occur after score commit but before job-record deletion.
        # Finish recovery/cleanup even when all submitted IDs are already scored.
        pending = active["compound_ids"]
    if not pending:
        ranking = 'Amber' if database.amber_keep_count(args) is not None and iteration > 0 else 'docking'
        print(f"Iteration {iteration}: no unscored final {ranking} top-{args.prime_gbsa_input_count} "
              "compounds ready to claim; checking on the next poll.")
        return
    action = "recovering saved prime_gbsa job" if active else "starting prime_gbsa"
    work_started(f"Iteration {iteration}: {action} for {len(pending)} compounds.", level=2)
    try:
        updated = c.run_iteration_prime_gbsa(
            iteration=iteration,
            args=args,
            iteration_dir=iteration_directory(args.output, iteration),
            glide_dir=stage_directory(args.output, iteration, 'glide'),
            compound_ids=pending,
        )
    except Exception as exc:
        # A status/download failure does not prove that the external job failed.
        if load_job(directory) is not None:
            raise
        rows = database.prime_gbsa_submission_rows(args.db_path, pending)
        pending = [cid for cid in pending if cid in rows and rows[cid][2] is None
                   and rows[cid][1] in (State.DOCKED, State.PRIME_GBSA_RUNNING)]
        if isinstance(exc, PrimeAtomTypingError):
            failed = sorted(set(pending) & set(exc.compound_ids))
            database.mark_prime_gbsa_atomtyping_failures(args.db_path, failed, str(exc))
            retryable = sorted(set(pending) - set(failed))
            if retryable:
                c.set_compound_state(
                    retryable, State.REFERENCE if iteration == 0 else State.DOCKED, failed_stage="prime_gbsa",
                    failure_message="Prime batch failed; no confirmed atomtyping error for this compound",
                    db_path=args.db_path,
                )
            print(f"Iteration {iteration}: {exc}; {len(failed)} marked failed, "
                  f"{len(retryable)} remain retryable.")
            return
        c.set_compound_state(
            pending,
            State.REFERENCE if iteration == 0 else State.DOCKED,
            failed_stage="prime_gbsa",
            failure_message=str(exc),
            db_path=args.db_path,
        )
        raise

    if updated is None:
        record = load_job(directory)
        # A compound-queue pass already prints its active/waiting summary.
        report = progress if record and record.get('mode') in ('compound_queue_v1', 'native_batch_v1') else print
        report(f"Iteration {iteration}: prime_gbsa submitted/running; results will be recovered on a later worker run.")
        return

    # IDs that were selected but produced no GBSA score become retryable docked compounds.
    with c.open_sqlite(args.db_path) as conn:
        missing = [
            row[0] for row in conn.execute(
                f"SELECT id FROM compound WHERE id IN ({','.join('?' for _ in pending)}) "
                "AND prime_gbsa_score IS NULL AND state=?",
                [*pending, State.PRIME_GBSA_RUNNING],
            )
        ] if pending else []
    if missing:
        c.set_compound_state(
            missing,
            State.REFERENCE if iteration == 0 else State.DOCKED,
            failed_stage="prime_gbsa",
            failure_message="No prime_gbsa score returned; retryable",
            db_path=args.db_path,
        )
    print(f"Iteration {iteration}: prime_gbsa DB scores updated={updated}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Independent PRIME_GBSA (Prime) stage")
    p.add_argument("project_toml", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--schrodinger", type=Path)
    ns = p.parse_args(argv)
    args = c.configure_project(ns.project_toml, ns.schrodinger)
    with database.stage_lock(args.db_path, "prime_gbsa"):
        _run_stage(args, ns.iteration)


if __name__ == "__main__":
    main()
