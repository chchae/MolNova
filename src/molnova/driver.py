#!/usr/bin/env python3
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from molnova import _core as c
from molnova import database
from molnova.stages._logging import WORK_STARTED

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
        text=True,
        bufsize=1,
        env=env,
    )

    pending = []
    active = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.strip() == WORK_STARTED:
                active = True
                pending.clear()
            elif active:
                print(f"[{stage:<8}] {line}", end="", flush=True)
            else:
                pending.append(line)
            if stop_event.is_set():
                break
    finally:
        if stop_event.is_set() and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            proc.wait()

    if proc.returncode != 0:
        for line in pending:
            print(f"[{stage:<8}] {line}", end="", flush=True)
    elif not active and not stop_event.is_set():
        reason = next((line.strip() for line in reversed(pending) if line.strip()),
                      "No eligible work.")
        print(f"[{stage:<8}] skip: {reason}", flush=True)
    return proc.returncode


def worker_loop(
    stage, module, project_toml, work_dir, poll_interval, stop_event, once,
    wake_event=None, after_run_event=None,
):
    cmd = [sys.executable, "-m", module, str(project_toml)]
    # Refill individual GBSA slots promptly; other workers keep their interval.
    if stage == "mmgbsa":
        poll_interval = min(poll_interval, 5)

    # In one-shot mode, evaluate the compounds persisted by this generation
    # attempt. The supervisor schedules subprocesses; workers still use SQLite.
    if once and wake_event is not None:
        wake_event.wait()

    while not stop_event.is_set():
        if wake_event is not None:
            wake_event.clear()
        try:
            rc = stream_process(stage, cmd, work_dir, stop_event)
        finally:
            if after_run_event is not None:
                after_run_event.set()
        if stop_event.is_set():
            return

        if rc != 0:
            if once:
                print(
                    f"[driver  ] {stage} exited with status {rc}; "
                    "stopping because --once was specified.",
                    flush=True,
                )
                return
            print(
                f"[driver  ] {stage} exited with status {rc}; "
                f"retrying after {poll_interval}s.",
                flush=True,
            )
        elif once:
            return

        if wake_event is None:
            stop_event.wait(poll_interval)
        else:
            wake_event.wait(poll_interval)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Concurrent supervisor for REINVENT/LigPrep/Glide/MM-GBSA/FEP workers"
    )
    p.add_argument("project_toml", type=Path)
    p.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds before restarting an idle/exited worker (default: 30)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run every stage worker once, then exit",
    )
    ns = p.parse_args(argv)

    if ns.poll_interval < 1:
        raise ValueError("--poll-interval must be >= 1")

    project_toml = ns.project_toml.expanduser().resolve()
    work_dir = project_toml.parent

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

    print("=" * 70)
    print("MODULAR LEAD-OPTIMIZATION DRIVER")
    print("=" * 70)
    print(f"Project        : {args.project}")
    print(f"TOML           : {project_toml}")
    print(f"SQLite DB      : {args.db_path}")
    print(f"Output         : {args.output}")
    print(f"Worker poll    : {ns.poll_interval} s")
    stages = build_stages(
        args.synthetic_feasibility_enabled,
        args.fep_enabled,
    )

    print("Stages         : " + ", ".join(stage for stage, _ in stages))
    print("Stop           : Ctrl-C")
    print()

    stop_event = threading.Event()
    synthetic_wake = threading.Event()

    def request_stop(signum=None, frame=None):
        if not stop_event.is_set():
            print("\n[driver  ] stopping workers...", flush=True)
            stop_event.set()
            synthetic_wake.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    threads = []
    for stage, module in stages:
        t = threading.Thread(
            target=worker_loop,
            args=(
                stage,
                module,
                project_toml,
                work_dir,
                ns.poll_interval,
                stop_event,
                ns.once,
                synthetic_wake if stage == "synthetic" else None,
                synthetic_wake if stage == "generate" else None,
            ),
            name=stage,
            daemon=False,
        )
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        request_stop()
        for t in threads:
            t.join()
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

    print("[driver  ] finished.")


if __name__ == "__main__":
    main()
