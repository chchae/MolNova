#!/usr/bin/env python3
import argparse
from pathlib import Path
import sqlite3
from molnova import _core as c


def choose_iteration(args, requested):
    if requested is not None:
        return requested
    with c.open_sqlite(args.db_path) as conn:
        max_iter = conn.execute(
            "SELECT COALESCE(MAX(iteration),0) FROM compound"
        ).fetchone()[0]
    return 1 if max_iter == 0 else max_iter + 1


def main(argv=None):
    p = argparse.ArgumentParser(description="REINVENT4/LibInvent generation stage")
    p.add_argument("project_toml", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--schrodinger", type=Path)
    ns = p.parse_args(argv)

    args = c.configure_project(ns.project_toml, ns.schrodinger)
    iteration = choose_iteration(args, ns.iteration)

    with c.open_sqlite(args.db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM compound WHERE iteration=?", (iteration,)
        ).fetchone()[0]
    if n:
        print(f"Iteration {iteration} already has {n} compounds; generation skipped.")
        return

    run_dir = args.output / f"iter{iteration}" / "reinvent"
    run_dir.mkdir(parents=True, exist_ok=True)

    if iteration == 1:
        model = args.libinvent_prior
        print("Iteration 1: sampling original LibInvent prior.")
    else:
        previous = iteration - 1
        with c.open_sqlite(args.db_path) as conn:
            n_prev_gbsa = conn.execute(
                """
                SELECT COUNT(*)
                FROM compound
                WHERE iteration=? AND gbsa_score IS NOT NULL
                """,
                (previous,),
            ).fetchone()[0]

        if n_prev_gbsa < args.gbsa_elite_count:
            print(
                f"Iteration {iteration} not ready: iteration {previous} has "
                f"only {n_prev_gbsa}/{args.gbsa_elite_count} MM-GBSA results."
            )
            return

        elites = c.select_global_elites(
            target_iteration=iteration,
            best_count=args.gbsa_elite_count,
        )
        if len(elites) < args.gbsa_elite_count:
            raise RuntimeError(
                f"Need {args.gbsa_elite_count} previous MM-GBSA elites; only {len(elites)} available."
            )
        _, train_file, valid_file, usable = c.write_elite_libinvent_training_files(
            elites, args.libinvent_scaffold_smiles, run_dir
        )
        if usable < 2:
            raise RuntimeError("Too few usable GBSA elites for LibInvent TL.")
        model = c.run_libinvent_elite_transfer_learning(
            run_dir=run_dir,
            prior_file=args.libinvent_prior,
            train_file=train_file,
            valid_file=valid_file,
            epochs=args.elite_tl_epochs,
        )

    generated = c.run_libinvent_sampling(
        run_dir=run_dir,
        libinvent_prior=model,
        scaffold_file=args.libinvent_scaffold_file,
        sample_size=args.sample_size,
    )
    inserted = c.insert_generated(
        csv_file=generated,
        iteration=iteration,
        target_count=args.target_count,
    )
    print(f"Iteration {iteration}: inserted {inserted} novel compounds.")


if __name__ == "__main__":
    main()
