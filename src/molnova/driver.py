#!/usr/bin/env python3
import argparse
import fcntl
import os
import signal
import subprocess
import select
import sys
import time
from pathlib import Path

from molnova import _core as c
from molnova import database
from molnova.stages._logging import WORK_STARTED, format_elapsed

BASE_STAGES = [
    ("generate", "molnova.stages.generate"),
    ("ligprep", "molnova.stages.ligprep"),
    ("glide", "molnova.stages.glide"),
    ("mmgbsa", "molnova.stages.mmgbsa"),
]


def build_stages(synthetic_feasibility_enabled, fep_enabled):
    stages = [BASE_STAGES[0]]
    if synthetic_feasibility_enabled:
        stages.append(("synthetic", "molnova.stages.synthetic_feasibility"))
    stages.extend(BASE_STAGES[1:])
    if fep_enabled:
        stages.append(("fep", "molnova.stages.fep"))
    return stages


def stream_process(stage, cmd, cwd, stop_event):
    """Stream actual work and failures; summarize each successful idle poll."""
    env = os.environ.copy()
    env["MOLNOVA_SUPERVISED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )

    pending = []
    active = False

    def emit(line):
        nonlocal active
        if line.strip() == WORK_STARTED:
            active = True
            pending.clear()
        elif active:
            print(f"[{stage:<8}] {line}", end="", flush=True)
        else:
            pending.append(line)

    try:
        assert proc.stdout is not None
        # Read bytes so buffered readline cannot hide output or block shutdown.
        buffer = b""
        while not stop_event.is_set():
            ready, _, _ = select.select([proc.stdout], [], [], 0.2)
            if not ready:
                continue
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            buffer += chunk
            lines = buffer.split(b"\n")
            buffer = lines.pop()
            for line in lines:
                emit(line.decode("utf-8", errors="replace") + "\n")
        if buffer:
            emit(buffer.decode("utf-8", errors="replace"))
    finally:
        if stop_event.is_set() and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        else:
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()

    if proc.returncode != 0:
        for line in pending:
            print(f"[{stage:<8}] {line}", end="", flush=True)
    elif not active and not stop_event.is_set():
        reason = next((line.strip() for line in reversed(pending) if line.strip()),
                      "No eligible work.")
        print(f"[{stage:<8}] skip: {reason}", flush=True)
    return proc.returncode


class StopFlag:
    """Signal-controlled cancellation for the single-thread supervisor."""

    def __init__(self):
        self.stopped = False

    def set(self):
        self.stopped = True

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.stopped:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 0.2))


def choose_iteration(args, requested=None):
    """Select once; every worker receives the same explicit iteration."""
    if requested is not None:
        if not 1 <= requested <= args.max_iteration:
            raise ValueError(f"Iteration must be between 1 and {args.max_iteration}.")
        return requested
    with database.connect(args.db_path) as conn:
        iterations = [row[0] for row in conn.execute(
            "SELECT DISTINCT iteration FROM compound WHERE iteration>0 ORDER BY iteration"
        )]
    for iteration in iterations:
        if database.iteration_completion_reason(
            args.db_path, iteration, args.target_count, args.gbsa_input_count, 0
        ) is not None:
            return iteration
    following = max(iterations, default=0) + 1
    return following if following <= args.max_iteration else None


def external_completion_reason(args, iteration, stage):
    """Do not advance past live or uncertain native work, even with stale scores."""
    if stage not in {"ligprep", "glide", "mmgbsa"}:
        return None
    from molnova import schrodinger_guard as guard
    from molnova.ligprep_recovery import load_job as load_ligprep
    from molnova.prime_async import load_job as load_prime

    directory = args.output / f"iter{iteration}"
    try:
        if stage == "ligprep" and load_ligprep(directory / "ligprep") is not None:
            return "LigPrep submission still requires recovery"
        if stage == "mmgbsa" and load_prime(directory / "mmgbsa") is not None:
            return "MM-GBSA queue still requires recovery"
        if stage == "glide":
            from molnova.glide_recovery import load_record
            for record_path in (directory / "glide").glob("*/glide_job.json"):
                record = load_record(record_path.parent)
                if not record.get("job_id"):
                    return "Glide launch identity remains uncertain"
                # Already downloaded records were verified terminal by the worker.
                if not record.get("downloaded"):
                    details = guard.job_details(args.schrodinger, record["job_id"])
                    if any(job.get("status") not in guard.TERMINAL for job in details):
                        return f"saved Glide job still active: {record['job_id']}"
        jobs = guard.active_jobs(args.schrodinger)
    except Exception as exc:
        return f"cannot verify JobServer completion: {exc}"
    directory = directory.resolve()
    for job in jobs:
        launch = job.get("spec", {}).get("launchParams", {}).get("launchDirectory")
        belongs = (launch and Path(launch).resolve().is_relative_to(directory)) or (
            str(directory) + "/" in job.get("commandLine", "")
        )
        if belongs:
            return f"external job still active: {job['jobId']}"
    return None


def supervise_iteration(args, project_toml, iteration, stages, poll_interval, stop, once=False):
    """Run one worker at a time; exit only after this iteration's final stage."""
    for stage, module in stages:
        started = time.monotonic()
        interval = min(poll_interval, 5) if stage == "mmgbsa" else poll_interval
        cmd = [sys.executable, "-m", module, str(project_toml), "--iteration", str(iteration)]
        while not stop.is_set():
            rc = stream_process(stage, cmd, project_toml.parent, stop)
            if stop.is_set():
                return False
            reason = f"worker exited with status {rc}" if rc else database.stage_completion_reason(
                args.db_path, iteration, stage, args.target_count, args.gbsa_input_count,
                args.synthetic_feasibility_enabled,
            )
            if reason is None:
                reason = external_completion_reason(args, iteration, stage)
            if reason is None:
                print(f"[driver  ] iteration {iteration}: {stage} complete "
                      f"({time.monotonic() - started:.1f} s including recovery/polling).", flush=True)
                break
            print(f"[driver  ] iteration {iteration}: {stage} pending: {reason}.", flush=True)
            if once:
                return False
            stop.wait(interval)
    return not stop.is_set()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Single-thread supervisor for one iteration of sequential stage workers"
    )
    p.add_argument("project_toml", type=Path)
    p.add_argument("--iteration", type=int, help="Iteration to run; default: oldest unfinished, or next new iteration")
    p.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds before restarting an idle/exited worker (default: 30)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Attempt each stage once in order; stop at the first unfinished stage",
    )
    ns = p.parse_args(argv)

    if ns.poll_interval < 1:
        raise ValueError("--poll-interval must be >= 1")

    project_toml = ns.project_toml.expanduser().resolve()

    # Configure once so the DB exists and project name is known.
    args = c.configure_project(project_toml)

    # One driver per project DB. Prevent accidental duplicate supervisors.
    lock_path = database.lock_path(args.db_path, "driver")
    lock_file = lock_path.open("a")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise SystemExit(
            f"Another driver appears to be running for {args.project}: {lock_path}"
        )

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()

    previous_handlers = {}
    try:
        iteration = choose_iteration(args, ns.iteration)
        if iteration is None:
            print("[driver  ] All configured iterations are complete.")
            return
        stages = build_stages(args.synthetic_feasibility_enabled, args.fep_enabled)
        print("=" * 70)
        print("SEQUENTIAL SINGLE-ITERATION DRIVER")
        print(f"Project        : {args.project}")
        print(f"TOML           : {project_toml}")
        print(f"SQLite DB      : {args.db_path}")
        print(f"Output         : {args.output}")
        print(f"Iteration      : {iteration}")
        print(f"Worker poll    : {ns.poll_interval} s (MM-GBSA: at most 5 s)")
        print("Stages         : " + " -> ".join(stage for stage, _ in stages))
        print("Stop           : Ctrl-C", flush=True)
        stop = StopFlag()

        def request_stop(signum=None, frame=None):
            if not stop.is_set():
                print("\n[driver  ] stopping current worker...", flush=True)
                stop.set()

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        database.start_iteration_timer(args.db_path, iteration)
        completed = supervise_iteration(
            args, project_toml, iteration, stages, ns.poll_interval, stop, ns.once,
        )
        if completed:
            elapsed, estimated = database.finish_iteration_timer(args.db_path, iteration)
            qualifier = ' (estimated from first compound creation)' if estimated else ''
            print(f"[driver  ] iteration {iteration}: finished; "
                  f"total elapsed={format_elapsed(elapsed)}{qualifier}.", flush=True)
        else:
            print(f"[driver  ] iteration {iteration}: paused; restart to resume.", flush=True)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


if __name__ == "__main__":
    main()
