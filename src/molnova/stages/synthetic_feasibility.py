#!/usr/bin/env python3
"""Evaluate compounds with RDKit SA scores and the AiZynthFinder batch CLI."""

import argparse
import gzip
import json
import shutil
import subprocess
from pathlib import Path

from molnova import _core as c
from molnova import database
from molnova.stages._logging import work_started, timed_stage
from molnova.chemistry.sa_score import calculate_sa_score
from molnova.states import CompoundState as State


def choose_iteration(args, requested):
    if requested is not None:
        return requested
    with c.open_sqlite(args.db_path) as conn:
        row = conn.execute(
            """
            SELECT MIN(iteration)
            FROM compound
            WHERE iteration>0
              AND (sa_score IS NULL OR (
                  synthetic_feasibility IS NULL AND state IN (?, ?)
              ))
            """,
            (State.GENERATED, State.SYNTHETIC_RUNNING),
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
    missing_scores = database.missing_sa_scores(args.db_path, iteration)
    if missing_scores:
        work_started(f"Iteration {iteration}: calculating RDKit SA scores for {len(missing_scores)} compounds.")
    scores = [(cid, calculate_sa_score(smiles)) for cid, smiles in missing_scores]
    updated = database.store_sa_scores(args.db_path, scores)
    if updated:
        print(f"Iteration {iteration}: RDKit SA scores stored for {updated} compounds.")

    with c.open_sqlite(args.db_path) as conn:
        compounds = conn.execute(
            """
            SELECT id, smiles
            FROM compound
            WHERE iteration=? AND synthetic_feasibility IS NULL
              AND state IN (?, ?)
            ORDER BY id
            """,
            (iteration, State.GENERATED, State.SYNTHETIC_RUNNING),
        ).fetchall()
    if not compounds:
        return 0

    remote = getattr(args, "remote_aizynth", None)
    if not remote and (args.aizynth_config is None or not args.aizynth_config.is_file()):
        raise FileNotFoundError(f"AiZynthFinder config not found: {args.aizynth_config}")
    executable = shutil.which(args.aizynth_cli) if not remote else None
    if not remote and executable is None:
        raise FileNotFoundError(
            f"AiZynthFinder CLI not found: {args.aizynth_cli}; configure 'aizynth-cli' or PATH."
        )

    ids = [row[0] for row in compounds]
    claimed_ids = set(
        database.claim_compounds(
            args.db_path,
            ids,
            expected_state=(State.GENERATED, State.SYNTHETIC_RUNNING),
            claimed_state=State.SYNTHETIC_RUNNING,
        )
    )
    compounds = [row for row in compounds if row[0] in claimed_ids]
    if not compounds:
        return 0

    work_started(f"Iteration {iteration}: starting AiZynthFinder for {len(compounds)} compounds.")
    output_dir = args.output / f"iter{iteration}" / "synthetic_feasibility"
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles_file = output_dir / "targets.smi"
    result_file = output_dir / "aizynthfinder_results.json.gz"
    smiles_file.write_text(
        "".join(f"{smiles}\n" for _, smiles in compounds),
        encoding="utf-8",
    )

    try:
        if remote:
            from molnova.aizynth_remote import run_remote
            run_remote(smiles_file, result_file, remote, nproc=getattr(args, "aizynth_nproc", 8))
        else:
            subprocess.run(
                [
                    executable,
                    "--config",
                    str(args.aizynth_config),
                    "--smiles",
                    str(smiles_file),
                    "--output",
                    str(result_file),
                    "--nproc",
                    str(getattr(args, "aizynth_nproc", 8)),
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
                    SET synthetic_feasibility=?, state=?,
                        failed_stage=NULL, failure_message=NULL,
                        modified_at=CURRENT_TIMESTAMP
                    WHERE id=? AND state=?
                      AND synthetic_feasibility IS NULL
                    """,
                    (feasible, State.GENERATED, by_smiles[smiles], State.SYNTHETIC_RUNNING),
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
            State.GENERATED,
            failed_stage="synthetic_feasibility",
            failure_message=str(exc),
            db_path=args.db_path,
        )
        raise


def import_results(args, iteration, result_file):
    if iteration is None or iteration <= 0:
        raise RuntimeError("--import-results requires --iteration greater than zero.")
    results = _read_results(Path(result_file))
    with c.open_sqlite(args.db_path) as conn:
        compounds = conn.execute(
            "SELECT id, smiles FROM compound WHERE iteration=? ORDER BY id",
            (iteration,),
        ).fetchall()
    by_smiles = {}
    for compound_id, smiles in compounds:
        canonical = c.canonicalize_smiles(smiles)
        by_smiles.setdefault(canonical, []).append(compound_id)
    unknown = set(results) - set(by_smiles)
    if unknown:
        raise RuntimeError(f"AiZynthFinder results contain {len(unknown)} unknown iteration targets.")
    values = [
        (compound_id, feasible)
        for smiles, feasible in results.items()
        for compound_id in by_smiles[smiles]
    ]
    scores = [
        (compound_id, calculate_sa_score(smiles))
        for compound_id, smiles in database.missing_sa_scores(args.db_path, iteration)
    ]
    updated = database.import_synthetic_results(args.db_path, iteration, values, scores)
    print(
        f"Iteration {iteration}: imported {updated} new synthetic results; "
        f"validated {len(results)} targets, {sum(results.values())} solved."
    )
    return updated


@timed_stage("Synthetic feasibility stage")
def _run_stage(args, requested_iteration, result_file=None):
    if result_file is not None:
        return import_results(args, requested_iteration, result_file)
    iteration = choose_iteration(args, requested_iteration)
    if iteration is None:
        print("No compounds require SA scoring or synthetic feasibility evaluation.")
        return
    process_iteration(args, iteration)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate compounds with RDKit SA scores and AiZynthFinder"
    )
    parser.add_argument("project_toml", type=Path)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--import-results", type=Path,
                        help="Import AiZynthFinder JSON/JSON.gz results without repeating searches")
    ns = parser.parse_args(argv)
    args = c.configure_project(ns.project_toml)
    with database.stage_lock(args.db_path, "synthetic_feasibility"):
        _run_stage(args, ns.iteration, ns.import_results)


if __name__ == "__main__":
    main()
