import argparse
import sys
from pathlib import Path

from molnova import _core
from molnova.driver import main as driver_main
from molnova.stages import fep, generate, glide, ligprep, mmgbsa
from molnova.stages import synthetic_feasibility


def _project_args(ns):
    args = [str(ns.project_toml)]
    if getattr(ns, "iteration", None) is not None:
        args += ["--iteration", str(ns.iteration)]
    if getattr(ns, "schrodinger", None) is not None:
        args += ["--schrodinger", str(ns.schrodinger)]
    return args


def _status(project_toml):
    args = _core.configure_project(project_toml)
    with _core.open_sqlite(args.db_path) as conn:
        rows = conn.execute(
            """
            SELECT iteration, state, COUNT(*)
            FROM compound
            GROUP BY iteration, state
            ORDER BY iteration, state
            """
        ).fetchall()
        scores = conn.execute(
            """
            SELECT iteration,
                   COUNT(docking_score), COUNT(gbsa_score), COUNT(fep_score),
                   COUNT(synthetic_feasibility), SUM(synthetic_feasibility),
                   MIN(docking_score), MIN(gbsa_score), MIN(fep_score)
            FROM compound
            GROUP BY iteration
            ORDER BY iteration
            """
        ).fetchall()

    print(f"Project: {args.project}")
    print(f"Database: {args.db_path}")
    print()
    print("States")
    print("iteration  state              count")
    print("---------  -----------------  -----")
    for iteration, state, count in rows:
        print(f"{iteration:9d}  {state:17s}  {count:5d}")

    print()
    print("Scores")
    print("iteration docked gbsa fep synth synth_yes  best_dock best_gbsa best_fep")
    print("--------- ------ ---- --- ----- ---------  --------- --------- --------")
    for (
        iteration,
        n_dock,
        n_gbsa,
        n_fep,
        n_synth,
        n_synth_yes,
        best_dock,
        best_gbsa,
        best_fep,
    ) in scores:
        print(
            f"{iteration:9d} {n_dock:6d} {n_gbsa:4d} {n_fep:3d} "
            f"{n_synth:5d} {str(n_synth_yes or 0):>9}  "
            f"{str(best_dock):>9} {str(best_gbsa):>9} {str(best_fep):>8}"
        )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and not argv[0].startswith("-") and Path(argv[0]).suffix.lower() == ".toml":
        argv.insert(0, "run")
    parser = argparse.ArgumentParser(
        prog="molnova",
        description="Iterative molecular lead-optimization pipeline",
        epilog="A TOML path alone runs the supervisor: molnova input/egfr.toml",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one iteration with sequential stage workers")
    run.add_argument("project_toml", type=Path)
    run.add_argument("--poll-interval", type=int, default=30)
    run.add_argument("--once", action="store_true")
    run.add_argument("--iteration", type=int)

    for name in ("generate", "ligprep", "glide", "mmgbsa"):
        stage = sub.add_parser(name, help=f"Run {name} stage")
        stage.add_argument("project_toml", type=Path)
        stage.add_argument("--iteration", type=int)
        stage.add_argument("--schrodinger", type=Path)
        if name == "glide":
            stage.add_argument("--poll-interval", type=int, default=30)
            stage.add_argument("--completion-fraction", type=float, default=0.95)
            stage.add_argument("--tail-timeout", type=int)

    synthetic = sub.add_parser(
        "synthetic-feasibility",
        help="Evaluate generated compounds with AiZynthFinder",
    )
    synthetic.add_argument("project_toml", type=Path)
    synthetic.add_argument("--iteration", type=int)
    synthetic.add_argument("--import-results", type=Path)

    fep_parser = sub.add_parser(
        "fep", help="Export FEP candidates or import FEP scores"
    )
    fep_parser.add_argument("project_toml", type=Path)
    fep_parser.add_argument("--iteration", type=int)
    fep_parser.add_argument("--import-scores", type=Path)
    fep_parser.add_argument("--schrodinger", type=Path)

    status = sub.add_parser("status", help="Show project DB status")
    status.add_argument("project_toml", type=Path)

    ns = parser.parse_args(argv)

    if ns.command == "run":
        args = [str(ns.project_toml), "--poll-interval", str(ns.poll_interval)]
        if ns.iteration is not None:
            args += ["--iteration", str(ns.iteration)]
        if ns.once:
            args.append("--once")
        return driver_main(args)

    if ns.command == "status":
        return _status(ns.project_toml)

    if ns.command == "generate":
        return generate.main(_project_args(ns))

    if ns.command == "ligprep":
        return ligprep.main(_project_args(ns))

    if ns.command == "glide":
        args = _project_args(ns)
        args += [
            "--poll-interval",
            str(ns.poll_interval),
            "--completion-fraction",
            str(ns.completion_fraction),
        ]
        if ns.tail_timeout is not None:
            args += ["--tail-timeout", str(ns.tail_timeout)]
        return glide.main(args)

    if ns.command == "mmgbsa":
        return mmgbsa.main(_project_args(ns))

    if ns.command == "synthetic-feasibility":
        args = _project_args(ns)
        if ns.import_results is not None:
            args += ["--import-results", str(ns.import_results)]
        return synthetic_feasibility.main(args)

    if ns.command == "fep":
        args = _project_args(ns)
        if ns.import_scores is not None:
            args += ["--import-scores", str(ns.import_scores)]
        return fep.main(args)


if __name__ == "__main__":
    main()
