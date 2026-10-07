"""Retry Prime submission after temporary license checkout failures."""
from pathlib import Path
import subprocess
import time


def license_unavailable(text):
    text = text.lower()
    return any(message in text for message in (
        "insufficient licenses", "insufficient licences",
        "license could not be obtained", "license checkout failed",
        "license checkout failure", "unable to check out",
        "licensed number of users already reached", "all licenses are in use",
    ))


def _logs(directory):
    return {path: (path.read_text(errors="replace"), path.stat().st_mtime_ns)
            for path in directory.rglob("*.log") if path.name != "prime_license_retry.log"}


def _run_attempt(command, cwd):
    lines = []
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        return proc.wait(), "".join(lines)


def run_prime(command, cwd, *, wait_seconds=300, retries=3):
    command, cwd = [str(value) for value in command], Path(cwd)
    last_license_output = ""
    for attempt in range(retries + 1):
        before = _logs(cwd)
        old_outputs = {path: path.stat().st_mtime_ns for path in cwd.glob("*-out.maegz")}
        print("$", " ".join(command), flush=True)
        code, output = _run_attempt(command, cwd)
        for path, (text, modified) in _logs(cwd).items():
            previous, old_modified = before.get(path, ("", None))
            if modified == old_modified and text == previous:
                continue
            output += text[len(previous):] if text != previous and text.startswith(previous) else text
        # A rewritten identical log can keep the same timestamp on shared filesystems.
        # Carry forward a confirmed license failure only when this retry has no diagnostics.
        if attempt and not output.strip():
            output = last_license_output
        new_output = any(path.stat().st_mtime_ns != old_outputs.get(path)
                         for path in cwd.glob("*-out.maegz"))
        if code == 0 and new_output:
            return
        if not license_unavailable(output):
            if code:
                raise subprocess.CalledProcessError(code, command, output=output)
            return  # The existing result parser reports missing output.
        if attempt == retries:
            raise RuntimeError(f"Prime license unavailable after {retries + 1} attempts; compounds remain retryable.")
        last_license_output = output
        message = (
            f"Prime license unavailable: waiting {wait_seconds}s before "
            f"retry {attempt + 1}/{retries}."
        )
        print(message, flush=True)
        with (cwd / "prime_license_retry.log").open("a") as stream:
            stream.write(message + "\n")
        time.sleep(wait_seconds)
