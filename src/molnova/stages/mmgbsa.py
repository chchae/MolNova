#!/usr/bin/env python3
import argparse
from pathlib import Path
from molnova import _core as c


def eligible_iteration(args):
    with c.open_sqlite(args.db_path) as conn:
        iterations = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT iteration FROM compound "
                "WHERE iteration>0 AND docking_score IS NOT NULL ORDER BY iteration"
            )
        ]
        for iteration in iterations:
            top = conn.execute(
                """
                SELECT id,state,gbsa_score
                FROM compound
                WHERE iteration=? AND docking_score IS NOT NULL
                ORDER BY docking_score ASC,id ASC
                LIMIT ?
                """,
                (iteration, args.gbsa_input_count),
            ).fetchall()
            if any(
                gbsa_score is None and state in ("docked", "gbsa_running")
                for _, state, gbsa_score in top
            ):
                return iteration
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description="Independent Prime MM-GBSA stage")
    p.add_argument("project_toml", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--schrodinger", type=Path)
    ns = p.parse_args(argv)
    args = c.configure_project(ns.project_toml, ns.schrodinger)
    iteration = ns.iteration or eligible_iteration(args)
    if iteration is None:
        print("No current docking top-N compounds require MM-GBSA.")
        return

    with c.open_sqlite(args.db_path) as conn:
        top = conn.execute(
            """
            SELECT id,state
            FROM compound
            WHERE iteration=? AND docking_score IS NOT NULL
            ORDER BY docking_score ASC,id ASC
            LIMIT ?
            """,
            (iteration, args.gbsa_input_count),
        ).fetchall()
        pending = [
            cid for cid, state in top
            if state in ("docked", "gbsa_running")
            and conn.execute(
                "SELECT gbsa_score FROM compound WHERE id=?", (cid,)
            ).fetchone()[0] is None
        ]

    if not pending:
        print(f"Iteration {iteration}: current docking top-{args.gbsa_input_count} already has MM-GBSA.")
        return

    c.set_compound_state(pending, "gbsa_running")
    try:
        updated = c.run_iteration_mmgbsa(
            iteration=iteration,
            args=args,
            iteration_dir=args.output / f"iter{iteration}",
            glide_dir=args.output / f"iter{iteration}" / "glide",
        )
    except Exception as exc:
        c.set_compound_state(
            pending,
            "docked",
            failed_stage="gbsa",
            failure_message=str(exc),
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
            "docked",
            failed_stage="gbsa",
            failure_message="No MM-GBSA score returned; retryable",
        )
    print(f"Iteration {iteration}: MM-GBSA DB scores updated={updated}")


if __name__ == "__main__":
    main()
