#!/usr/bin/env python3
import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from molnova.paths import iteration_directory, stage_directory, stage_in_iteration
from molnova import _core as c
from molnova import database
from molnova.stages._logging import work_started


def choose_iteration(args, requested):
    if requested is not None:
        return requested
    return next((iteration for iteration in database.fep_candidate_iterations(args.db_path)
                 if database.fep_selection_pending(args.db_path, iteration, args.fep_input_count)), None)


def export_candidates(args, iteration):
    rows = database.fep_candidates(
        args.db_path, iteration, args.fep_input_count,
        target_count=getattr(args, 'target_count', 1),
        prime_count=getattr(args, 'gbsa_input_count', None),
        amber_keep_count=database.amber_keep_count(args))
    outdir = stage_directory(args.output, iteration, 'fep')
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / 'fep_candidates.tsv'
    fields = ['id', 'name', 'smiles', 'docking_score', 'amber_gbsa_score', 'gbsa_score',
              'fep_score', 'state', 'ranking_source', 'ranking_score']
    fd, temporary = tempfile.mkstemp(dir=outdir, prefix='candidates-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, outfile)
    finally:
        Path(temporary).unlink(missing_ok=True)
    source = rows[0]['ranking_source'] if rows else 'none'
    print(f'Iteration {iteration}: exported {len(rows)} FEP candidates; ranking={source}')
    print(f'Output: {outfile}')
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

    iterations = [iteration] if iteration is not None else database.fep_candidate_iterations(args.db_path)
    candidate_ids = set()
    for candidate_iteration in iterations:
        rows = database.fep_candidates(args.db_path, candidate_iteration, args.fep_input_count, for_import=True)
        candidate_ids.update(row['id'] for row in rows)
    updated = database.store_fep_scores(args.db_path, updates, candidate_ids)
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
        print("No scored compounds currently require FEP staging.")
        return
    for stage in ('generate', 'druglike', 'synthetic_feasibility', 'ligprep', 'glide', 'amber_gbsa', 'mmgbsa'):
        if database.stage_is_running(args.db_path, stage):
            raise RuntimeError(f'FEP waiting for {stage} worker')
    from molnova import amber_gbsa, prime_async, ligprep_recovery
    directory = iteration_directory(args.output, iteration)
    if (amber_gbsa.load_job(stage_in_iteration(directory, 'amber_gbsa')) is not None
            or prime_async.load_job(stage_in_iteration(directory, 'mmgbsa')) is not None
            or ligprep_recovery.load_job(stage_in_iteration(directory, 'ligprep')) is not None):
        raise RuntimeError('FEP waiting for saved external submissions to retire')
    from molnova.driver import external_completion_reason
    reason = external_completion_reason(args, iteration, 'glide')
    if reason:
        raise RuntimeError(f'FEP waiting: {reason}')
    work_started(f"Iteration {iteration}: exporting FEP candidates.")
    export_candidates(args, iteration)
    print("FEP+ execution is intentionally not automated yet; configure the FEP protocol first.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "FEP staging utility. Exports Prime/Amber/docking-ranked candidates now; "
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
