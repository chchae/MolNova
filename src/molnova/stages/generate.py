#!/usr/bin/env python3
import argparse
from pathlib import Path
import sqlite3
import csv
import os
import tempfile
import time
from molnova import _core as c
from molnova import database
from molnova.chemistry.scaffold import prepare_iteration_scaffold, log_iteration_mcs
from molnova.chemistry.descriptors import calculate_rdkit_descriptors_batch
from molnova.chemistry.druglike import filter_druglike
from molnova.stages._logging import work_started, timed_stage


def choose_iteration(args, requested):
    if requested is not None:
        if requested < 1 or requested > args.max_iteration:
            raise ValueError(
                f"Iteration must be between 1 and {args.max_iteration}."
            )
        return requested
    with c.open_sqlite(args.db_path) as conn:
        max_iter = conn.execute(
            "SELECT COALESCE(MAX(iteration),0) FROM compound"
        ).fetchone()[0]
        if max_iter == 0:
            return 1 if args.max_iteration >= 1 else None
        current_count = conn.execute(
            "SELECT COUNT(*) FROM compound WHERE iteration=?",
            (max_iter,),
        ).fetchone()[0]

    if current_count < args.target_count:
        return max_iter
    next_iteration = max_iter + 1
    return next_iteration if next_iteration <= args.max_iteration else None


def store_iteration_descriptors(args, iteration):
    compounds = database.missing_rdkit_descriptors(args.db_path, iteration)
    if not compounds:
        return 0
    cpus = getattr(args, 'sa_score_cpus', 1)
    work_started(f'Iteration {iteration}: calculating local RDKit properties (including SA score) '
                 f'for {len(compounds)} compounds (CPU limit={cpus}).')
    scores = calculate_rdkit_descriptors_batch(compounds, cpus)
    updated = database.store_rdkit_descriptors(args.db_path, scores)
    print(f'Iteration {iteration}: RDKit properties stored for {updated} compounds.', flush=True)
    return updated


def insert_druglike_batch(csv_file, iteration, target_count, db_path, cpus=1, *, combichem_progress=None):
    csv_file = Path(csv_file)
    generated = c.read_generated_smiles(csv_file)
    novel = database.novel_generated_smiles(db_path, generated)
    print(f'Already in DB      : {len(generated) - len(novel)}', flush=True)
    work_started(f'Iteration {iteration}: calculating RDKit properties for '
                 f'{len(novel)} valid novel molecules before druglikeness selection (CPU limit={cpus}).')
    properties = calculate_rdkit_descriptors_batch(
        [(i, smiles, None) for i, smiles in enumerate(novel)], cpus)
    rows = [dict(values, batch_index=i, smiles=novel[i],
                 qed=values['rdk_qed'], sa_score=values['rdk_sa_score'],
                 ro5_violations=values['rdk_ro5_violations']) for i, values in properties]
    selected = filter_druglike(rows)
    # Retain each batch's accepted/rejected inputs; indices are not SQLite IDs.
    report = csv_file.parent / f'druglike_selection-{time.time_ns()}.csv'
    fields = ['batch_index', 'smiles', *database.RDK_DESCRIPTOR_TYPES,
              'druglike_selected', 'druglike_reject_reason']
    fd, temporary = tempfile.mkstemp(dir=report.parent, prefix=report.name + '.')
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(selected)
        os.replace(temporary, report)
    finally:
        Path(temporary).unlink(missing_ok=True)
    progress = {'combichem_progress': combichem_progress} if combichem_progress is not None else {}
    inserted = database.insert_generated_candidates(db_path, iteration, target_count, selected, **progress)
    ro5_rejected = sum(row['druglike_reject_reason'] == 'ro5_violations' for row in selected)
    accepted = sum(row['druglike_selected'] for row in selected)
    print(f'Iteration {iteration}: druglikeness input={len(rows)}, RO5 rejected={ro5_rejected}, '
          f'QED rank rejected={len(rows) - ro5_rejected - accepted}, accepted={accepted}, '
          f'inserted={inserted}.', flush=True)
    print(f'Druglikeness report: {report}', flush=True)
    return inserted


def select_iteration_sa_compounds(args, iteration):
    selected = database.apply_sa_filter(
        args.db_path, iteration, args.target_count, getattr(args, 'sa_accepted_ratio', 0.75),
    )
    if selected:
        ratio, total, accepted = selected
        print(f'Iteration {iteration}: SA selection accepted {accepted}/{total} compounds '
              f'(ratio={ratio:g}, lowest SA scores first); {total - accepted} sa_rejected.', flush=True)
    else:
        print(f'Iteration {iteration}: SA selection deferred for incomplete generation '
              'or preserved because downstream work has already started.', flush=True)


def _run_combichem(args, iteration, directory):
    from molnova.chemistry import combichem
    work_started(f'Iteration {iteration}: starting local combichem generation.')
    plan = combichem.prepare_plan(args, iteration, directory)
    from molnova.chemistry.scaffold import _publish
    import json
    _publish(directory / 'plan.json', json.dumps(plan, indent=2) + '\n')
    print(f"Combichem: {len(plan['pools'])} attachment sites; "
          f"{[len(pool) for pool in plan['pools']]} sidechains; "
          f"{plan['total_combinations']} Cartesian combinations; "
          f"library={plan['library_path']}; weight={plan['selection_weight']}.", flush=True)
    total = int(plan['total_combinations'])
    start = plan['next_index']
    while database.iteration_compound_count(args.db_path, iteration) < args.target_count:
        if start >= total:
            count = database.iteration_compound_count(args.db_path, iteration)
            raise RuntimeError(f'Iteration {iteration}: combichem combinations exhausted at '
                               f'{count}/{args.target_count} compounds; generation remains incomplete.')
        stop = min(total, start + plan['batch_size'])
        generated = combichem.write_batch(plan, start, stop, directory)
        inserted = insert_druglike_batch(
            generated, iteration, args.target_count, args.db_path, getattr(args, 'sa_score_cpus', 1),
            combichem_progress=(start, stop),
        )
        print(f'Iteration {iteration}: combichem attempted {stop}/{total} combinations, '
              f'inserted={inserted}.', flush=True)
        start = stop
    store_iteration_descriptors(args, iteration)
    select_iteration_sa_compounds(args, iteration)


@timed_stage("Generate stage")
def _run_stage(args, requested_iteration):
    iteration = choose_iteration(args, requested_iteration)
    if iteration is None:
        print(f"Maximum iteration ({args.max_iteration}) reached.")
        return

    with c.open_sqlite(args.db_path) as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM compound WHERE iteration=?", (iteration,)
        ).fetchone()[0]
    remaining = max(0, args.target_count - n)
    if remaining == 0:
        log_iteration_mcs(args, iteration)
        store_iteration_descriptors(args, iteration)
        select_iteration_sa_compounds(args, iteration)
        print(
            f"Iteration {iteration} already has the target "
            f"{args.target_count} compounds; generation skipped."
        )
        return

    combichem = getattr(args, 'sidechain_library', None) is not None
    run_dir = args.output / f"iter{iteration}" / ('combichem' if combichem else 'reinvent')
    run_dir.mkdir(parents=True, exist_ok=True)

    if iteration == 1:
        prepare_iteration_scaffold(args, iteration, create=True)
        if combichem:
            return _run_combichem(args, iteration, run_dir)
        work_started("Iteration 1: starting LibInvent sampling.")
        model = args.libinvent_prior
        print("Iteration 1: sampling original LibInvent prior.")
    else:
        previous = iteration - 1
        # Also check earlier iterations when recovering a DB created by the old
        # early-generation policy. Explicit --iteration cannot bypass this gate.
        for completed_iteration in range(1, iteration):
            if database.amber_keep_count(args) is not None:
                from molnova.amber_gbsa import load_job as load_amber
                if (load_amber(args.output / f'iter{completed_iteration}/amber_gbsa') is not None
                        or database.stage_is_running(args.db_path, 'amber_gbsa')):
                    print(f'Iteration {iteration} not ready: preceding Amber work still requires recovery.')
                    return
            reason = database.iteration_completion_reason(
                args.db_path, completed_iteration, args.target_count,
                args.gbsa_input_count, args.gbsa_elite_count,
                database.amber_keep_count(args),
            )
            if reason is not None:
                print(f"Iteration {iteration} not ready: iteration "
                      f"{completed_iteration}: {reason}.")
                return

        work_started(f"Iteration {iteration}: starting " + (
            'elite-guided combichem generation.' if combichem else 'elite transfer learning and sampling.'))
        elites = database.iteration_gbsa_elites(
            args.db_path, previous, args.gbsa_elite_count,
        )
        if len(elites) < args.gbsa_elite_count:
            raise RuntimeError(
                f"Need {args.gbsa_elite_count} previous MM-GBSA elites; only {len(elites)} available."
            )
        prepare_iteration_scaffold(args, iteration, elites, create=True)
        if combichem:
            return _run_combichem(args, iteration, run_dir)
        _, train_file, valid_file, usable = c.write_elite_libinvent_training_files(
            elites, args.libinvent_scaffold_smiles, run_dir,
            prior_file=args.libinvent_prior,
            remote=getattr(args, "remote_reinvent", None),
        )
        if usable < 2:
            raise RuntimeError(
                f"Too few usable GBSA elites for LibInvent TL ({usable}; need at least 2). "
                f"See {run_dir / 'elite_skipped.tsv'} for unsupported-token exclusions."
            )
        model = c.run_libinvent_elite_transfer_learning(
            run_dir=run_dir,
            prior_file=args.libinvent_prior,
            train_file=train_file,
            valid_file=valid_file,
            epochs=args.elite_tl_epochs,
            remote=getattr(args, "remote_reinvent", None),
        )

    generated = c.run_libinvent_sampling(
        run_dir=run_dir,
        libinvent_prior=model,
        scaffold_file=args.libinvent_scaffold_file,
        sample_size=args.sample_size,
        remote=getattr(args, "remote_reinvent", None),
        remote_model=iteration == 1,
    )
    inserted = c.insert_generated(
        csv_file=generated,
        iteration=iteration,
        target_count=args.target_count,
        db_path=args.db_path,
        cpus=getattr(args, 'sa_score_cpus', 1),
    )
    store_iteration_descriptors(args, iteration)
    select_iteration_sa_compounds(args, iteration)
    if inserted == 0:
        raise RuntimeError(
            f"Iteration {iteration} remains below target-count "
            f"({n}/{args.target_count}) and no novel druglike compounds were inserted."
        )
    print(f"Iteration {iteration}: inserted {inserted} novel compounds.")


def main(argv=None):
    p = argparse.ArgumentParser(description="REINVENT4/LibInvent generation stage")
    p.add_argument("project_toml", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--schrodinger", type=Path)
    p.add_argument("--elite-mcs", help="Generation SMARTS override (takes precedence over TOML)")
    ns = p.parse_args(argv)

    overrides = {'elite_mcs': ns.elite_mcs} if ns.elite_mcs is not None else {}
    args = c.configure_project(ns.project_toml, ns.schrodinger, **overrides)
    with database.stage_lock(args.db_path, "generate"):
        _run_stage(args, ns.iteration)


if __name__ == "__main__":
    main()
