"""Persist Prime submissions so stage workers can exit while jobs run."""
import json
import hashlib
import re
import os
import socket
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

from molnova.schrodinger_retry import _run_attempt, license_unavailable


class PrimeAtomTypingError(RuntimeError):
    """A submitted compound could not be assigned Prime force-field types."""

    def __init__(self, compound_ids):
        self.compound_ids = ([compound_ids] if isinstance(compound_ids, int)
                             else sorted(set(compound_ids)))
        self.compound_id = self.compound_ids[0] if len(self.compound_ids) == 1 else None
        titles = ", ".join(f"CMPID_{cid}" for cid in self.compound_ids)
        super().__init__(f"{titles}: Prime atom typing failed; "
                         "see mmgbsa_input.Prime.log and mmgbsa_input.err.log")


def _check_single_compound_atomtyping(text, compound_ids):
    # Batch-wide diagnostics cannot identify which entry failed in a mixed batch.
    # Only classify a single submitted ID explicitly named in the diagnostics.
    if len(compound_ids) != 1:
        return
    compound_id = compound_ids[0]
    if not re.search(rf"\bCMPID_{compound_id}\b", text):
        return
    lowered = text.lower()
    if any(marker in lowered for marker in (
        "cannot atom type structure", "problem in atomtyping structure",
        "failure running atom typer",
    )):
        raise PrimeAtomTypingError(compound_id)


def atomtyping_failed_ids(text, compound_ids):
    """Associate explicit atomtyping errors with per-entry Prime log sections."""
    allowed = set(compound_ids)
    failed = set()
    for section in re.split(r"(?=^\s*Prime-MMGBSA:)|(?=^Entry:\s*\d+)", text,
                            flags=re.MULTILINE):
        if not re.match(r"\s*(Prime-MMGBSA:|Entry:)", section):
            continue
        ids = {int(cid) for cid in re.findall(r"\bCMPID_(\d+)\b", section)}
        if len(ids) == 1 and ids <= allowed and any(marker in section.lower() for marker in (
            "cannot atom type structure", "problem in atomtyping structure",
            "failure running atom typer",
        )):
            failed.update(ids)
    if not failed:
        try:
            _check_single_compound_atomtyping(text, compound_ids)
        except PrimeAtomTypingError as exc:
            failed.update(exc.compound_ids)
    return sorted(failed)


def _log_signature(path):
    return [path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()]


def _current_diagnostics(directory, record):
    baseline = record.get("log_snapshot", {})
    logs = []
    for path in directory.glob("*.log"):
        if path.name == "prime_license_retry.log":
            continue
        text = path.read_text(errors="replace")
        named_job = re.search(r"^JobId\s*:\s*(\S+)", text, re.MULTILINE)
        if named_job and named_job.group(1) != record.get("job_id"):
            continue
        if (record.get("job_id") and record["job_id"] in text
                or "log_snapshot" in record and baseline.get(path.name) != _log_signature(path)):
            logs.append(text)
    return "\nPrime-MMGBSA: LOG_BOUNDARY\n".join(logs)


def persist_atomtyping_failures(directory, db_path):
    """Record only unscored failures, after successful batch scores are imported."""
    from molnova import database
    record = load_job(directory) or {}
    failed = record.get("atomtyping_failed_ids", [])
    if failed:
        database.mark_gbsa_atomtyping_failures(db_path, failed,
                                              str(PrimeAtomTypingError(failed)))


def job_file(directory):
    return Path(directory) / "prime_job.json"


def load_job(directory):
    path = job_file(directory)
    return json.loads(path.read_text()) if path.exists() else None


def _save(directory, record):
    path = job_file(directory)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record))
    temporary.replace(path)


class PrimeReconciliationPending(RuntimeError):
    """A mismatched saved job cannot yet be safely removed."""


def _job_details(record, directory):
    jsc = Path(record["command"][0]).parent / "jsc"
    result = subprocess.run([str(jsc), "info", "--json", record["job_id"]],
                            cwd=directory, check=True, capture_output=True, text=True)
    jobs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    return jobs


def _job_status(record, directory):
    return next(job["status"] for job in _job_details(record, directory)
                if job["jobId"] == record["job_id"])


def reconcile_saved_job(directory, db_path, iteration):
    """Retire orphaned job files while preserving current results and live jobs."""
    from molnova import database
    from molnova.stages._logging import work_started
    record = load_job(directory)
    if record is None:
        return None
    ids = set(record["compound_ids"])
    rows = database.mmgbsa_submission_rows(db_path, ids)
    current = {cid for cid, row in rows.items() if row[0] == iteration}
    if ids and current == ids:
        return record
    if current:
        raise PrimeReconciliationPending(
            f"Iteration {iteration}: saved Prime job mixes current and obsolete compound IDs; "
            "files retained for reconciliation.")
    if record.get("mode") == "compound_queue_v1":
        raise PrimeReconciliationPending(
            f"Iteration {iteration}: obsolete compound queue retained for job reconciliation.")
    if record.get("job_id"):
        jobs = _job_details(record, directory)
        status = next(job["status"] for job in jobs if job["jobId"] == record["job_id"])
        if any(job.get("status") not in {"DONE", "FAILED", "CANCELED", "STOPPED"} for job in jobs):
            raise PrimeReconciliationPending(
                f"Iteration {iteration}: obsolete Prime job {record['job_id']} is {status}; "
                "waiting for it to end before cleaning files.")
    elif record.get("launching"):
        raise PrimeReconciliationPending(
            f"Iteration {iteration}: obsolete Prime submission has no saved JobId; "
            "files retained because launch status is unknown.")
    directory = Path(directory)
    archived = directory.with_name(f"{directory.name}.stale-{time.time_ns()}")
    directory.rename(archived)
    directory.mkdir()
    work_started(f"Iteration {iteration}: archived obsolete MM-GBSA files to {archived.name}; "
                 "using current database compounds.")
    return None


def _launch_process_alive(directory, record):
    """Inspect the launch host, including surviving submission clients."""
    if record.get("launch_host", socket.gethostname()) != socket.gethostname():
        raise PrimeReconciliationPending("Prime launch occurred on another host; local process absence is insufficient")
    directory = Path(directory).resolve()
    input_path = directory / Path(record["command"][1]).name
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            words = (proc / "cmdline").read_bytes().split(b"\0")
            for word in words:
                value = word.decode(errors="replace")
                if value.startswith("/") and Path(value).resolve() == input_path:
                    return True
        except FileNotFoundError:
            continue  # The process exited during inspection.
        except PermissionError as exc:
            raise PrimeReconciliationPending("Cannot inspect a submission process on the launch host") from exc
    return False


def _launch_history(directory, record):
    """Verified jsc flags include terminal jobs; Slurm alone is insufficient."""
    from molnova.schrodinger_guard import job_details, job_kind
    suite = Path(record["command"][0]).parent
    jobs = {}
    # Shared filesystem aliases can differ from the native launch directory.
    for launch in dict.fromkeys([str(Path(directory).absolute()), str(Path(directory).resolve())]):
        cmd = [str(suite / "jsc"), "list", "--any-status", "--launch-dir", launch, "--id-only"]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
        if result.returncode:
            message = (result.stdout + result.stderr).strip()
            if result.returncode == 1 and message == "No jobs matching your search criteria were found.":
                continue
            result.check_returncode()
        for job_id in result.stdout.splitlines():
            job_id = job_id.strip()
            if job_id and job_id not in jobs:
                for job in job_details(suite, job_id, directory):
                    if job_kind(job) == "mmgbsa":
                        jobs[job["jobId"]] = job
    return list(jobs.values())


def reconcile_interrupted_submission(directory, record):
    """Recover a missing JobId, or release a positively unsubmitted launch."""
    directory = Path(directory)
    if _launch_process_alive(directory, record):
        raise PrimeReconciliationPending("Prime submission client is still running; waiting for its JobId")
    # Record every emitted JobId immediately, before the launch client exits.
    # Legacy logs also carry an authoritative job identity if newly written.
    ids = set()
    for path in directory.glob("*.log"):
        if path.name == "prime_license_retry.log":
            continue
        if record.get("log_snapshot", {}).get(path.name) == _log_signature(path):
            continue
        ids.update(re.findall(r"^JobId\s*:\s*(\S+)", path.read_text(errors="replace"),
                              re.MULTILINE | re.IGNORECASE))
    if len(ids) > 1:
        raise PrimeReconciliationPending("Multiple Prime JobIds in current launch logs; reconciliation is ambiguous")
    jobs = _launch_history(directory, record)
    if ids:
        job_id = ids.pop()
        # Verify that a logged job exists before adopting it.
        from molnova.schrodinger_guard import job_details
        job_details(Path(record["command"][0]).parent, job_id, directory)
    else:
        parents = [job for job in jobs if not job.get("parentJobId")]
        if len(parents) > 1 or jobs and not parents:
            raise PrimeReconciliationPending("Prime launch history is ambiguous; submissions withheld")
        job_id = parents[0]["jobId"] if parents else None
    if job_id:
        record.update(job_id=job_id, launching=False)
        _save(directory, record)
        print(f"Recovered interrupted Prime submission JobId: {job_id}.", flush=True)
        return True
    started = record.get("launch_started_at", job_file(directory).stat().st_mtime)
    if time.time() - started < 60:
        raise PrimeReconciliationPending("Prime launch registration grace period has not elapsed")
    # Unidentified calculation artifacts prevent claiming that nothing launched.
    if any(directory.glob("*-out.maegz")) or any(
        path.name != "prime_license_retry.log" and path.stat().st_size
        and record.get("log_snapshot", {}).get(path.name) != _log_signature(path)
        for path in directory.glob("*.log")
    ):
        raise PrimeReconciliationPending("Unidentified Prime calculation artifacts remain; submissions withheld")
    archived = directory / f"prime_job.unsubmitted-{time.time_ns()}.json"
    archived.write_text(json.dumps(record))
    record.update(launching=False, retry_at=0, attempt=max(0, record.get("attempt", 1) - 1),
                  recovered_unsubmitted_at=time.time())
    _save(directory, record)
    print("Interrupted Prime launch had no client, JobServer job or calculation artifacts; "
          "released for guarded resubmission.", flush=True)
    return False


def submit_or_poll(command, directory, compound_ids, *, wait_seconds=300, retries=3,
                   on_completed_subjobs=None, submission_fence=None):
    """Return True only after a terminal job's outputs have been downloaded."""
    directory = Path(directory)
    record = load_job(directory)
    if record is None:
        record = {"command": [str(v) for v in command],
                  "compound_ids": list(compound_ids), "attempt": 0,
                  "job_id": None, "retry_at": 0, "launching": False}
    command = record["command"]
    jsc = Path(command[0]).parent / "jsc"
    if record.get("launching") and not record.get("job_id"):
        try:
            if not reconcile_interrupted_submission(directory, record):
                return False  # Queue releases the slot and refills under its submission fence.
        except Exception as exc:
            raise PrimeReconciliationPending(f"Prime submission was interrupted; recovery pending: {exc}") from exc
    if record["job_id"]:
        jobs = _job_details(record, directory)
        parent = next(job for job in jobs if job["jobId"] == record["job_id"])
        status = parent["status"]
        if on_completed_subjobs is not None:
            on_completed_subjobs(record, jobs)
        if any(job.get("status") not in {"DONE", "FAILED", "CANCELED", "STOPPED"} for job in jobs):
            print(f"Prime job {record['job_id']}: {status}; checking on next worker run.")
            return False
        subprocess.run([str(jsc), "download", "--cwd", record["job_id"]],
                       cwd=directory, check=True, capture_output=True, text=True)
        diagnostics = _current_diagnostics(directory, record)
        failed = atomtyping_failed_ids(diagnostics, record["compound_ids"])
        all_failed = "all entries failed" in diagnostics.lower()
        if (status == "DONE" and any(directory.glob("*-out.maegz"))
                and not all_failed and set(failed) != set(record["compound_ids"])):
            record["atomtyping_failed_ids"] = failed
            _save(directory, record)
            from molnova.stages._logging import job_elapsed, format_elapsed
            elapsed = job_elapsed(parent)
            if elapsed is not None:
                print(f"Prime MM-GBSA calculation completed; elapsed={format_elapsed(elapsed)}.", flush=True)
            return True
        if failed:
            job_file(directory).unlink()
            raise PrimeAtomTypingError(failed)
        if not license_unavailable(diagnostics):
            job_file(directory).unlink()
            raise RuntimeError(f"Prime job {record['job_id']} ended with {status}: {diagnostics[-4000:]}")
        record["job_id"] = None
        record["retry_at"] = time.time() + wait_seconds
        record["launching"] = False
        _save(directory, record)
    if record["launching"]:
        raise RuntimeError("Prime submission was interrupted before its JobId was saved; "
                           "reconcile prime_job.json with JobServer before resubmitting.")
    if record["retry_at"] > time.time():
        return False
    if record["attempt"] > retries:
        job_file(directory).unlink(missing_ok=True)
        raise RuntimeError(f"Prime license unavailable after {retries + 1} attempts; compounds remain retryable.")
    from molnova.schrodinger_guard import SubmissionBlocked
    try:
        with submission_fence() if submission_fence is not None else nullcontext():
            record["log_snapshot"] = {path.name: _log_signature(path)
                                      for path in directory.glob("*.log")}
            record["launching"] = True
            record["launch_host"] = socket.gethostname()
            record["launch_started_at"] = time.time()
            record["attempt"] += 1
            _save(directory, record)
            print("$", " ".join(command), flush=True)
            def checkpoint(line):
                match = re.search(r"^JobId:\s*(\S+)", line, re.IGNORECASE)
                if match:
                    record.update(job_id=match.group(1), launching=False)
                    _save(directory, record)
            code, output = _run_attempt(command, directory, checkpoint)
    except SubmissionBlocked as exc:
        print(f'Prime submission waiting: {exc}.', flush=True)
        return False
    match = re.search(r"^JobId:\s*(\S+)", output, re.MULTILINE | re.IGNORECASE)
    if match:
        record.update(job_id=match.group(1), launching=False)
        _save(directory, record)
        return False
    if license_unavailable(output):
        record.update(launching=False, retry_at=time.time() + wait_seconds)
        _save(directory, record)
        return False
    if code:
        job_file(directory).unlink()
        diagnostics = output + "\n" + _current_diagnostics(directory, record)
        failed = atomtyping_failed_ids(diagnostics, record["compound_ids"])
        if failed:
            raise PrimeAtomTypingError(failed)
        raise subprocess.CalledProcessError(code, command, output=output)
    raise RuntimeError("Prime submission returned no JobId; reconcile prime_job.json before resubmitting.")
