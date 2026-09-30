#!/usr/bin/env python3
import argparse
from pathlib import Path
import sqlite3
import time
from molnova import _core as c


def query_groups(iteration):
    groups = {}
    with c.open_sqlite(c.DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id,name,smiles,similar_to,similarity
            FROM compound
            WHERE iteration=? AND state IN ('ligprepped','glide_running')
              AND docking_score IS NULL
              AND similar_to IS NOT NULL
            ORDER BY similar_to,id
            """, (iteration,)
        ).fetchall()
    for cid, name, smiles, refid, sim in rows:
        groups.setdefault(refid, []).append((cid,name,smiles,sim))
    return groups


def mark_group(iteration, compounds, state, failed_stage=None, message=None):
    c.set_compound_state(
        [x[0] for x in compounds],
        state,
        failed_stage=failed_stage,
        failure_message=message,
    )


def merge_current_best(args, glide_root, merger):
    files = sorted(glide_root.glob("ref_*/best_poses.maegz"))
    if files:
        c.merge_best_pose_files(
            args.schrodinger, merger, files,
            glide_root / "best_poses.maegz", glide_root
        )


def process_completed_job(iteration, args, job, extractor, merger, glide_root):
    score_file, best_file = c.extract_scores_and_best_poses(
        args.schrodinger, extractor, job["output_file"], job["group_dir"]
    )
    updated = c.update_scores(score_file)
    with c.open_sqlite(args.db_path) as conn:
        scored = {r[0] for r in conn.execute(
            "SELECT id FROM compound WHERE iteration=? AND docking_score IS NOT NULL",
            (iteration,),
        )}
    ids = [x[0] for x in job["compounds"]]
    failed = [cid for cid in ids if cid not in scored]
    if failed:
        c.set_compound_state(failed, "failed", failed_stage="glide", failure_message="No Glide pose")
    merge_current_best(args, glide_root, merger)
    print(f"Reference {job['reference_id']} completed: DB updated={updated}, no-pose={len(failed)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Independent asynchronous Glide stage")
    p.add_argument("project_toml", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--poll-interval", type=int, default=30)
    p.add_argument("--completion-fraction", type=float, default=0.95)
    p.add_argument("--tail-timeout", type=int)
    p.add_argument("--schrodinger", type=Path)
    ns = p.parse_args(argv)
    args = c.configure_project(ns.project_toml, ns.schrodinger)
    args.poll_interval = ns.poll_interval
    args.completion_fraction = ns.completion_fraction
    if ns.tail_timeout is not None:
        args.tail_timeout = ns.tail_timeout

    iteration = ns.iteration or c.find_iteration_for_state("glide")
    if iteration is None:
        print("No iteration requires Glide.")
        return

    # Assign nearest reference for any still-undocked compounds.
    c.assign_reference_compounds(iteration, args)
    groups = query_groups(iteration)
    if not groups:
        print(f"Iteration {iteration}: no Glide-pending compounds.")
        return

    ligprep_file = args.output / f"iter{iteration}" / "ligprep" / "ligprep_all.maegz"
    if not ligprep_file.exists():
        raise FileNotFoundError(f"LigPrep output not found: {ligprep_file}")

    glide_root = args.output / f"iter{iteration}" / "glide"
    glide_root.mkdir(parents=True, exist_ok=True)
    group_ligand_files = c.split_ligprep_by_reference(
        args.schrodinger, ligprep_file, groups, glide_root
    )
    ref_extractor = c.create_reference_extractor(glide_root)
    score_extractor = c.create_score_extractor(glide_root)
    merger = c.create_best_pose_merger(glide_root)
    core_matcher = c.create_core_atom_matcher(glide_root)

    jobs = []
    for refid, compounds in sorted(groups.items()):
        ligand_file = group_ligand_files.get(refid)
        if ligand_file is None:
            mark_group(iteration, compounds, "failed", failed_stage="glide", message="No LigPrep structures for reference group")
            continue
        refname = c.get_reference_name(refid)
        gdir = glide_root / f"ref_{refid}"
        gdir.mkdir(parents=True, exist_ok=True)
        ref_file = gdir / "_reference.maegz"
        output_file = gdir / "glide_constrained_lib.maegz"
        log_file = gdir / "glide_constrained.log"

        job = dict(reference_id=refid, reference_name=refname,
                   group_dir=gdir, reference_file=ref_file,
                   output_file=output_file, log_file=log_file,
                   input_count=len(compounds), compounds=compounds)

        # A previously submitted job may have completed while this driver was not running.
        if output_file.exists() and output_file.stat().st_size > 0:
            process_completed_job(iteration, args, job, score_extractor, merger, glide_root)
            continue

        # Existing non-failed log without output: assume the JobServer/SLURM job is still running.
        if log_file.exists() and not c.glide_log_failed(log_file):
            mark_group(iteration, compounds, "glide_running")
            jobs.append(job)
            continue

        c.extract_reference_pose(
            args.schrodinger, ref_extractor, args.reference_poses,
            refname, ref_file, gdir
        )
        core_atoms = None
        if args.mcs_smarts:
            core_atoms = c.get_core_atoms_from_reference(
                args.schrodinger, core_matcher, ref_file, args.mcs_smarts, gdir
            )
        inp = c.write_constrained_glide_input(
            gdir, args.grid, ligand_file, ref_file,
            args.mcs_smarts, core_atoms
        )
        submitted = c.submit_glide(args.schrodinger, inp, gdir, args.host)
        job.update(submitted)
        mark_group(iteration, compounds, "glide_running")
        jobs.append(job)

    if not jobs:
        print("No running Glide jobs remain.")
        return

    total = len(jobs)
    terminal = 0
    tail_started = None
    start = time.monotonic()
    pending = list(jobs)

    while pending:
        rest = []
        for job in pending:
            if job["output_file"].exists() and job["output_file"].stat().st_size > 0:
                process_completed_job(iteration, args, job, score_extractor, merger, glide_root)
                terminal += 1
                continue
            if c.glide_log_failed(job["log_file"]):
                mark_group(iteration, job["compounds"], "failed", failed_stage="glide", message="Glide job failed")
                terminal += 1
                print(f"Reference {job['reference_id']} failed.")
                continue
            rest.append(job)
        pending = rest
        if not pending:
            break

        frac = terminal / total if total else 1.0
        now = time.monotonic()
        if tail_started is None and frac >= args.completion_fraction:
            tail_started = now
            print(f"Tail phase started: {terminal}/{total} terminal, timeout={args.tail_timeout}s")
        if tail_started is not None and now - tail_started >= args.tail_timeout:
            print(f"Tail timeout reached; leaving {len(pending)} jobs running for a later 03-glide.py invocation.")
            break
        print(f"Glide: {len(pending)} running, {terminal}/{total} terminal")
        time.sleep(args.poll_interval)

    merge_current_best(args, glide_root, merger)


if __name__ == "__main__":
    main()
