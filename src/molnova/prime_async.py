"""Persist Prime submissions so stage workers can exit while jobs run."""
import json
import hashlib
import re
import subprocess
import time
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


def submit_or_poll(command, directory, compound_ids, *, wait_seconds=300, retries=3):
    """Return True only after a terminal job's outputs have been downloaded."""
    directory = Path(directory)
    record = load_job(directory)
    if record is None:
        record = {"command": [str(v) for v in command],
                  "compound_ids": list(compound_ids), "attempt": 0,
                  "job_id": None, "retry_at": 0, "launching": False}
    command = record["command"]
    jsc = Path(command[0]).parent / "jsc"
    if record["job_id"]:
        result = subprocess.run([str(jsc), "info", "--json", record["job_id"]],
                                cwd=directory, check=True, capture_output=True, text=True)
        jobs = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        job = next(j for j in jobs if j["jobId"] == record["job_id"])
        if job["status"] not in {"DONE", "FAILED", "CANCELED", "STOPPED"}:
            print(f"Prime job {record['job_id']}: {job['status']}; checking on next worker run.")
            return False
        subprocess.run([str(jsc), "download", "--cwd", record["job_id"]],
                       cwd=directory, check=True, capture_output=True, text=True)
        diagnostics = _current_diagnostics(directory, record)
        failed = atomtyping_failed_ids(diagnostics, record["compound_ids"])
        all_failed = "all entries failed" in diagnostics.lower()
        if (job["status"] == "DONE" and any(directory.glob("*-out.maegz"))
                and not all_failed and set(failed) != set(record["compound_ids"])):
            record["atomtyping_failed_ids"] = failed
            _save(directory, record)
            return True
        if failed:
            job_file(directory).unlink()
            raise PrimeAtomTypingError(failed)
        if not license_unavailable(diagnostics):
            job_file(directory).unlink()
            raise RuntimeError(f"Prime job {record['job_id']} ended with {job['status']}: {diagnostics[-4000:]}")
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
    record["log_snapshot"] = {path.name: _log_signature(path)
                              for path in directory.glob("*.log")}
    record["launching"] = True
    record["attempt"] += 1
    _save(directory, record)
    print("$", " ".join(command), flush=True)
    code, output = _run_attempt(command, directory)
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
