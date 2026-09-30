#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from molnova import _core as c


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


def import_scores(args, score_file):
    updates = []
    with score_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "id" not in reader.fieldnames or "fep_score" not in reader.fieldnames:
            raise RuntimeError("FEP score TSV must contain columns: id, fep_score")
        for row in reader:
            if not row.get("fep_score"):
                continue
            updates.append((float(row["fep_score"]), int(row["id"])))
    if not updates:
        print("No FEP scores to import.")
        return 0
    with c.open_sqlite(args.db_path) as conn:
        conn.executemany(
            """
            UPDATE compound
            SET fep_score=?, state='fep_done', failed_stage=NULL, failure_message=NULL,
                modified_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            updates,
        )
        conn.commit()
    print(f"Imported FEP scores: {len(updates)}")
    return len(updates)


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

    if ns.import_scores is not None:
        import_scores(args, ns.import_scores.expanduser().resolve())
        return

    iteration = choose_iteration(args, ns.iteration)
    if iteration is None:
        print("No GBSA-ranked compounds currently require FEP staging.")
        return
    export_candidates(args, iteration)
    print("FEP+ execution is intentionally not automated yet; configure the FEP protocol first.")


if __name__ == "__main__":
    main()
