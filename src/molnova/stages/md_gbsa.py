"""Desmond GPU / Prime CPU ensemble stage, independent of the supervisor."""
import argparse
from pathlib import Path
from molnova import _core, database, md_gbsa
from molnova.stages._logging import timed_stage, work_started


@timed_stage('MD-GBSA worker')
def _run_stage(args, iteration):
    work_started(f'Iteration {iteration}: recovering or starting Desmond/Prime MD-GBSA.')
    md_gbsa.run(args,iteration)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project_toml',type=Path)
    parser.add_argument('--iteration',type=int,required=True)
    parser.add_argument('--schrodinger',type=Path)
    ns=parser.parse_args(argv)
    args=_core.configure_project(ns.project_toml,ns.schrodinger)
    with database.stage_lock(args.db_path,'md_gbsa'):
        _run_stage(args,ns.iteration)


if __name__=='__main__':
    main()
