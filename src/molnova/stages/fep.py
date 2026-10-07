#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from molnova import _core as c
from molnova import database
from molnova.stages._logging import work_started


def choose_iteration(args, requested):
    if requested is not None:
        return requested
    with c.open_sqlite(args.db_path) as conn:
        iterations = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT iteration FROM compound "
                "WHERE iteration>0 AND gbsa_score IS NOT NULL ORDER BY iteration"
            )
        ]
        for iteration in iterations:
            top = conn.execute(
                """
                SELECT id,state,fep_score
                FROM compound
                WHERE iteration=? AND gbsa_score IS NOT NULL
                ORDER BY gbsa_score ASC,id ASC
                LIMIT ?
                """,
                (iteration, args.fep_input_count),
            ).fetchall()
            if any(fep_score is None for _, _, fep_score in top):
                return iteration
    return None


def export_candidates(args, iteration):
    outdir = args.output / f"iter{iteration}" / "fep"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "fep_candidates.tsv"
    with c.open_sqlite(args.db_path) as conn:
        rows = conn.execute(
            """
            SELECT id,name,smiles,docking_score,gbsa_score,fep_score,state
            FROM compound
            WHERE iteration=? AND gbsa_score IS NOT NULL
            ORDER BY gbsa_score ASC,id ASC
            LIMIT ?
            """,
            (iteration, args.fep_input_count),
        ).fetchall()
    with outfile.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id","name","smiles","docking_score","gbsa_score","fep_score","state"])
        w.writerows(rows)
    print(f"Iteration {iteration}: exported {len(rows)} FEP candidates")
    print(f"Output: {outfile}")
    return outfile


def import_scores(args, score_file, iteration=None):
    updates = {}
    with score_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "id" not in reader.fieldnames or "fep_score" not in reader.fieldnames:
            raise RuntimeError("FEP score TSV must contain columns: id, fep_score")
        for row in reader:
            if not row.get("fep_score"):
                continue
            compound_id = int(row["id"])
            score = float(row["fep_score"])
            if not math.isfinite(score):
                raise ValueError(f"FEP score for compound {compound_id} must be finite")
            if compound_id in updates:
                raise ValueError(f"Duplicate FEP score for compound {compound_id}")
            updates[compound_id] = score
    if not updates:
        print("No FEP scores to import.")
        return 0

    with c.open_sqlite(args.db_path) as conn:
        if iteration is None:
            iterations = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT iteration FROM compound "
                    "WHERE iteration>0 AND gbsa_score IS NOT NULL"
                )
            ]
        else:
            iterations = [iteration]

        candidate_ids = set()
        for candidate_iteration in iterations:
            candidate_ids.update(
                row[0]
                for row in conn.execute(
                    """
                    SELECT id FROM compound
                    WHERE iteration=? AND gbsa_score IS NOT NULL
                    ORDER BY gbsa_score ASC, id ASC
                    LIMIT ?
                    """,
                    (candidate_iteration, args.fep_input_count),
                )
            )

        invalid_ids = set(updates) - candidate_ids
        if invalid_ids:
            raise ValueError(
                "FEP scores include IDs outside the selected GBSA candidate set: "
                + ", ".join(str(value) for value in sorted(invalid_ids))
            )

        updated = 0
        for compound_id, score in updates.items():
            cursor = conn.execute(
                """
                UPDATE compound
                SET fep_score=?, state='fep_done', failed_stage=NULL,
                    failure_message=NULL, modified_at=CURRENT_TIMESTAMP
                WHERE id=? AND iteration>0 AND gbsa_score IS NOT NULL
                """,
                (score, compound_id),
            )
            updated += cursor.rowcount
        conn.commit()
    print(f"Imported FEP scores: {updated}")
    return updated


def _run_stage(args, ns):
    if ns.import_scores is not None:
        work_started("Importing FEP scores.")
        import_scores(
            args,
            ns.import_scores.expanduser().resolve(),
            iteration=ns.iteration,
        )
        return

    iteration = choose_iteration(args, ns.iteration)
    if iteration is None:
        print("No GBSA-ranked compounds currently require FEP staging.")
        return
    work_started(f"Iteration {iteration}: exporting FEP candidates.")
    export_candidates(args, iteration)
    print("FEP+ execution is intentionally not automated yet; configure the FEP protocol first.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "FEP staging utility. Exports GBSA-ranked candidates now; "
            "can later import id/fep_score TSV results without assuming an FEP+ protocol."
        )
    )
    p.add_argument("project_toml", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--import-scores", type=Path)
    p.add_argument("--schrodinger", type=Path)
    ns = p.parse_args(argv)
    args = c.configure_project(ns.project_toml, ns.schrodinger)
    with database.stage_lock(args.db_path, "fep"):
        _run_stage(args, ns)


if __name__ == "__main__":
    main()
