#!/usr/bin/env python3
import argparse
from pathlib import Path
from molnova import _core as c
from molnova import database
from molnova.stages._logging import work_started
from molnova.states import CompoundState as State


def eligible_iteration(args):
    with c.open_sqlite(args.db_path) as conn:
        iterations = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT iteration FROM compound "
                "WHERE iteration>0 AND docking_score IS NOT NULL ORDER BY iteration"
            )
        ]
    for iteration in iterations:
        if database.mmgbsa_candidates(args.db_path, iteration, args.gbsa_input_count):
            return iteration
    return None


def _run_stage(args, requested_iteration):
    iteration = requested_iteration or eligible_iteration(args)
    if iteration is None:
        print("No current docking top-N compounds require MM-GBSA.")
        return

    pending = database.mmgbsa_candidates(args.db_path, iteration, args.gbsa_input_count)

    if not pending:
        print(f"Iteration {iteration}: current docking top-{args.gbsa_input_count} already has MM-GBSA.")
        return

    pending = database.claim_compounds(
        args.db_path,
        pending,
        expected_state=(State.DOCKED, State.GBSA_RUNNING),
        claimed_state=State.GBSA_RUNNING,
    )
    if not pending:
        print(f"Iteration {iteration}: top-N compounds were claimed by another worker.")
        return
    work_started(f"Iteration {iteration}: starting MM-GBSA for {len(pending)} compounds.")
    try:
        updated = c.run_iteration_mmgbsa(
            iteration=iteration,
            args=args,
            iteration_dir=args.output / f"iter{iteration}",
            glide_dir=args.output / f"iter{iteration}" / "glide",
            compound_ids=pending,
        )
    except Exception as exc:
        c.set_compound_state(
            pending,
            State.DOCKED,
            failed_stage="gbsa",
            failure_message=str(exc),
            db_path=args.db_path,
        )
        raise

    # IDs that were selected but produced no GBSA score become retryable docked compounds.
    with c.open_sqlite(args.db_path) as conn:
        missing = [
            row[0] for row in conn.execute(
                f"SELECT id FROM compound WHERE id IN ({','.join('?' for _ in pending)}) "
                "AND gbsa_score IS NULL",
                pending,
            )
        ] if pending else []
    if missing:
        c.set_compound_state(
            missing,
            State.DOCKED,
            failed_stage="gbsa",
            failure_message="No MM-GBSA score returned; retryable",
            db_path=args.db_path,
        )
    print(f"Iteration {iteration}: MM-GBSA DB scores updated={updated}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Independent Prime MM-GBSA stage")
    p.add_argument("project_toml", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--schrodinger", type=Path)
    ns = p.parse_args(argv)
    args = c.configure_project(ns.project_toml, ns.schrodinger)
    with database.stage_lock(args.db_path, "mmgbsa"):
        _run_stage(args, ns.iteration)


if __name__ == "__main__":
    main()
