#!/usr/bin/env python3
"""Evaluate generated compounds with the AiZynthFinder batch CLI."""

import argparse
import gzip
import json
import shutil
import subprocess
from pathlib import Path

from molnova import _core as c
from molnova import database


def choose_iteration(args, requested):
    if requested is not None:
        return requested
    with c.open_sqlite(args.db_path) as conn:
        row = conn.execute(
            """
            SELECT MIN(iteration)
            FROM compound
            WHERE iteration>0
              AND synthetic_feasibility IS NULL
              AND state IN ('generated','synthetic_running')
            """
        ).fetchone()
    return row[0] if row else None


def _read_results(result_file):
    opener = gzip.open if result_file.suffix == ".gz" else open
    with opener(result_file, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("AiZynthFinder output has no table 'data' rows.")

    results = {}
    for row in rows:
        target = row.get("target")
        canonical = c.canonicalize_smiles(target) if target else None
        if canonical is None:
            raise RuntimeError("AiZynthFinder returned a result with invalid target SMILES.")
        if "number_of_solved_routes" in row:
            feasible = int(row["number_of_solved_routes"] or 0) > 0
        elif "is_solved" in row:
            value = row["is_solved"]
            feasible = value if isinstance(value, bool) else str(value).lower() == "true"
        else:
            raise RuntimeError(
                "AiZynthFinder output lacks number_of_solved_routes and is_solved."
            )
        if canonical in results:
            raise RuntimeError(f"AiZynthFinder returned duplicate target {canonical}.")
        results[canonical] = int(feasible)
    return results


def process_iteration(args, iteration):
    if not args.synthetic_feasibility_enabled:
        raise RuntimeError(
            "Enable synthetic-feasibility-enabled in the project TOML to run this stage."
        )
    if args.aizynth_config is None or not args.aizynth_config.is_file():
        raise FileNotFoundError(f"AiZynthFinder config not found: {args.aizynth_config}")
    executable = shutil.which(args.aizynth_cli)
    if executable is None:
        raise FileNotFoundError(
            f"AiZynthFinder CLI not found: {args.aizynth_cli}; configure 'aizynth-cli' or PATH."
        )

    with c.open_sqlite(args.db_path) as conn:
        compounds = conn.execute(
            """
            SELECT id, smiles
            FROM compound
            WHERE iteration=? AND synthetic_feasibility IS NULL
              AND state IN ('generated','synthetic_running')
            ORDER BY id
            """,
            (iteration,),
        ).fetchall()
    if not compounds:
        return 0

    ids = [row[0] for row in compounds]
    claimed_ids = set(
        database.claim_compounds(
            args.db_path,
            ids,
            expected_state=("generated", "synthetic_running"),
            claimed_state="synthetic_running",
        )
    )
    compounds = [row for row in compounds if row[0] in claimed_ids]
    if not compounds:
        return 0

    output_dir = args.output / f"iter{iteration}" / "synthetic_feasibility"
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles_file = output_dir / "targets.smi"
    result_file = output_dir / "aizynthfinder_results.json.gz"
    smiles_file.write_text(
        "".join(f"{smiles}\n" for _, smiles in compounds),
        encoding="utf-8",
    )

    try:
        subprocess.run(
            [
                executable,
                "--config",
                str(args.aizynth_config),
                "--smiles",
                str(smiles_file),
                "--output",
                str(result_file),
            ],
            cwd=output_dir,
            check=True,
        )
        results = _read_results(result_file)
        by_smiles = {
            c.canonicalize_smiles(smiles): compound_id
            for compound_id, smiles in compounds
        }
        if set(results) != set(by_smiles):
            missing = set(by_smiles) - set(results)
            unexpected = set(results) - set(by_smiles)
            raise RuntimeError(
                "AiZynthFinder result targets do not match the submitted batch; "
                f"missing={len(missing)}, unexpected={len(unexpected)}."
            )

        with c.open_sqlite(args.db_path) as conn:
            for smiles, feasible in results.items():
                cursor = conn.execute(
                    """
                    UPDATE compound
                    SET synthetic_feasibility=?, state='generated',
                        failed_stage=NULL, failure_message=NULL,
                        modified_at=CURRENT_TIMESTAMP
                    WHERE id=? AND state='synthetic_running'
                      AND synthetic_feasibility IS NULL
                    """,
                    (feasible, by_smiles[smiles]),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(
                        f"Could not persist synthetic feasibility for compound "
                        f"{by_smiles[smiles]}."
                    )
        feasible_count = sum(results.values())
        print(
            f"Iteration {iteration}: AiZynthFinder evaluated {len(results)} compounds; "
            f"solved routes found for {feasible_count}."
        )
        return len(results)
    except Exception as exc:
        c.set_compound_state(
            [row[0] for row in compounds],
            "generated",
            failed_stage="synthetic_feasibility",
            failure_message=str(exc),
            db_path=args.db_path,
        )
        raise


def _run_stage(args, requested_iteration):
    iteration = choose_iteration(args, requested_iteration)
    if iteration is None:
        print("No generated compounds require synthetic feasibility evaluation.")
        return
    process_iteration(args, iteration)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate generated compounds with AiZynthFinder"
    )
    parser.add_argument("project_toml", type=Path)
    parser.add_argument("--iteration", type=int)
    ns = parser.parse_args(argv)
    args = c.configure_project(ns.project_toml)
    with database.stage_lock(args.db_path, "synthetic_feasibility"):
        _run_stage(args, ns.iteration)


if __name__ == "__main__":
    main()