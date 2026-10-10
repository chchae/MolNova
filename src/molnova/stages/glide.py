#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path
import sqlite3
import time
from molnova import _core as c
from molnova import database
from molnova.stages._logging import work_started
from molnova.states import CompoundState as State


def prepare_glide_files(glide_root, ligprep_file):
    """Remove results from another LigPrep input before inspecting job outputs.

    Keep matching-input files so asynchronous jobs survive worker restarts.
    Legacy outputs without a fingerprint are treated as stale.
    """
    digest = hashlib.sha256()
    with ligprep_file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    signature = digest.hexdigest()
    marker = glide_root / "ligprep_input.sha256"
    if marker.exists() and marker.read_text().strip() == signature:
        return

    removed = 0
    for group in glide_root.glob("ref_*"):
        if not group.is_dir():
            continue
        artifacts = set(group.glob("glide_constrained*"))
        artifacts.update(group / name for name in (
            "best_poses.maegz", "docking_scores.tsv", "_reference.maegz",
            "ligprep_group.maegz",
        ))
        for path in artifacts:
            if path.is_file():
                path.unlink()
                removed += 1
    for name in ("best_poses.maegz", "best_poses.pending.maegz", "reference_map.tsv"):
        path = glide_root / name
        if path.is_file():
            path.unlink()
            removed += 1
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(signature + "\n")
    temporary.replace(marker)
    if removed:
        print(f"Removed {removed} old Glide files for the current LigPrep input.")


def query_groups(iteration, db_path):
    groups = {}
    with c.open_sqlite(db_path) as conn:
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


def mark_group(db_path, compounds, state, failed_stage=None, message=None):
    c.set_compound_state(
        [x[0] for x in compounds],
        state,
        failed_stage=failed_stage,
        failure_message=message,
        db_path=db_path,
    )


def merge_current_best(args, glide_root, merger):
    files = sorted(glide_root.glob("ref_*/best_poses.maegz"))
    if files:
        pending_file = glide_root / "best_poses.pending.maegz"
        c.merge_best_pose_files(
            args.schrodinger, merger, files,
            pending_file, glide_root
        )
        pending_file.replace(glide_root / "best_poses.maegz")


def process_completed_job(iteration, args, job, extractor, merger, glide_root):
    score_file, best_file = c.extract_scores_and_best_poses(
        args.schrodinger, extractor, job["output_file"], job["group_dir"]
    )
    # Publish complete poses before making the group's scores eligible for GBSA.
    merge_current_best(args, glide_root, merger)
    updated = c.update_scores(score_file, args.db_path)
    with c.open_sqlite(args.db_path) as conn:
        scored = {r[0] for r in conn.execute(
            "SELECT id FROM compound WHERE iteration=? AND docking_score IS NOT NULL",
            (iteration,),
        )}
    ids = [x[0] for x in job["compounds"]]
    failed = [cid for cid in ids if cid not in scored]
    if failed:
        c.set_compound_state(
            failed,
            "failed",
            failed_stage="glide",
            failure_message="No Glide pose",
            db_path=args.db_path,
        )
    print(f"Reference {job['reference_id']} completed: DB updated={updated}, no-pose={len(failed)}")


def _run_stage(args, ns):
    args.poll_interval = ns.poll_interval
    args.completion_fraction = ns.completion_fraction
    if ns.tail_timeout is not None:
        args.tail_timeout = ns.tail_timeout

    iteration = ns.iteration or c.find_iteration_for_state("glide", args.db_path)
    if iteration is None:
        print("No iteration requires Glide.")
        return

    reason = database.upstream_completion_reason(
        args.db_path, iteration, getattr(args, "target_count", 1), "glide",
    )
    if reason:
        print(f"Iteration {iteration}: Glide waiting: {reason}.")
        return

    # Assign nearest reference for any still-undocked compounds.
    c.assign_reference_compounds(iteration, args)
    groups = query_groups(iteration, args.db_path)

    claimed_groups = {}
    for reference_id, compounds in groups.items():
        claimed_ids = set(
            database.claim_compounds(
                args.db_path,
                [compound[0] for compound in compounds],
                expected_state=("ligprepped", "glide_running"),
                claimed_state="glide_running",
            )
        )
        if claimed_ids:
            claimed_groups[reference_id] = [
                compound for compound in compounds if compound[0] in claimed_ids
            ]
    groups = claimed_groups
    if not groups:
        print(f"Iteration {iteration}: no Glide-pending compounds.")
        return

    work_started(f"Iteration {iteration}: processing Glide for {len(groups)} reference groups.")
    ligprep_file = args.output / f"iter{iteration}" / "ligprep" / "ligprep_all.maegz"
    if not ligprep_file.exists():
        raise FileNotFoundError(f"LigPrep output not found: {ligprep_file}")

    glide_root = args.output / f"iter{iteration}" / "glide"
    glide_root.mkdir(parents=True, exist_ok=True)
    prepare_glide_files(glide_root, ligprep_file)
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
            mark_group(args.db_path, compounds, "failed", failed_stage="glide", message="No LigPrep structures for reference group")
            continue
        refname = c.get_reference_name(refid, args.db_path)
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

        if c.glide_log_no_poses(log_file):
            mark_group(args.db_path, compounds, State.FAILED, failed_stage="glide", message="Completed Glide job produced no poses")
            continue

        # Existing non-failed log without output: assume the JobServer/SLURM job is still running.
        if log_file.exists() and not c.glide_log_failed(log_file):
            mark_group(args.db_path, compounds, "glide_running")
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
        submitted = c.submit_glide(
            args.schrodinger, inp, gdir, args.host,
            cpus=getattr(args, "glide_cpus", None),
        )
        job.update(submitted)
        mark_group(args.db_path, compounds, "glide_running")
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
            if c.glide_log_no_poses(job["log_file"]):
                mark_group(args.db_path, job["compounds"], State.FAILED, failed_stage="glide", message="Completed Glide job produced no poses")
                terminal += 1
                print(f"Reference {job['reference_id']} completed without poses.")
                continue
            if c.glide_log_failed(job["log_file"]):
                mark_group(args.db_path, job["compounds"], "failed", failed_stage="glide", message="Glide job failed")
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
    with database.stage_lock(args.db_path, "glide"):
        _run_stage(args, ns)


if __name__ == "__main__":
    main()
