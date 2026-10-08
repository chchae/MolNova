"""Persist Prime submissions so stage workers can exit while jobs run."""
import json
import re
import subprocess
import time
from pathlib import Path

from molnova.schrodinger_retry import _run_attempt, license_unavailable


class PrimeAtomTypingError(RuntimeError):
    """A submitted compound could not be assigned Prime force-field types."""

    def __init__(self, compound_id):
        self.compound_id = compound_id
        super().__init__(f"CMPID_{compound_id}: Prime atom typing failed; "
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
        diagnostics = "\n".join(p.read_text(errors="replace") for p in directory.glob("*.log"))
        if not license_unavailable(diagnostics):
            try:
                # Use only logs naming this job, so old logs cannot classify a
                # later submission as a permanent structure failure.
                current = "\n".join(
                    text for path in directory.glob("*.log")
                    if record["job_id"] in (text := path.read_text(errors="replace"))
                )
                _check_single_compound_atomtyping(current, record["compound_ids"])
            except PrimeAtomTypingError:
                job_file(directory).unlink()
                raise
        if job["status"] == "DONE" and any(directory.glob("*-out.maegz")):
            return True
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
        _check_single_compound_atomtyping(output, record["compound_ids"])
        raise subprocess.CalledProcessError(code, command, output=output)
    raise RuntimeError("Prime submission returned no JobId; reconcile prime_job.json before resubmitting.")
