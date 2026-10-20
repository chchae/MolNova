"""Evaluate and freeze drug likeness across the complete generated iteration."""
import argparse
import csv
import os
from pathlib import Path
import tempfile

from molnova import _core as c, database
from molnova.chemistry.druglike import filter_druglike
from molnova.stages.generate import store_iteration_descriptors
from molnova.stages._logging import work_started, timed_stage


@timed_stage('Drug likeness worker')
def _run_stage(args, iteration):
    if iteration < 1:
        raise ValueError('Drug likeness requires iteration >= 1')
    database.validate_generation_population(args.db_path, iteration, args.target_count)
    if not database.uses_raw_generation(args.db_path, iteration):
        print(f'Iteration {iteration}: retaining legacy drug likeness/SA selection.')
        return
    if database.iteration_compound_count(args.db_path, iteration) < args.target_count or database.stage_is_running(args.db_path, 'generate'):
        print(f'Iteration {iteration}: drug likeness waiting for complete generation.')
        return
    work_started(f'Iteration {iteration}: assessing drug likeness of all generated database compounds.')
    store_iteration_descriptors(args, iteration)
    if database.druglike_completion_reason(args.db_path, iteration) is not None:
        selections = filter_druglike(database.druglike_population(args.db_path, iteration))
        database.apply_druglike_selection(args.db_path, iteration, args.target_count,
                                          selections, getattr(args, 'sa_accepted_ratio', 1.0))
    rows = database.druglike_report(args.db_path, iteration)
    folder = args.output / f'iter{iteration}/druglike'
    folder.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=folder, prefix='selection-')
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['id', 'smiles', 'rdk_qed', 'rdk_sa_score', 'rdk_ro5_violations', 'passed', 'reason'])
            writer.writerows(rows)
        os.replace(temporary, folder / 'druglike_selection.csv')
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f'Iteration {iteration}: drug likeness passed={sum(row[-2] for row in rows)}/{len(rows)}.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Iteration-wide drug likeness selection')
    parser.add_argument('project_toml', type=Path)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--schrodinger', type=Path)
    ns = parser.parse_args(argv)
    args = c.configure_project(ns.project_toml, ns.schrodinger)
    with database.stage_lock(args.db_path, 'druglike'):
        _run_stage(args, ns.iteration)


if __name__ == '__main__':
    main()
