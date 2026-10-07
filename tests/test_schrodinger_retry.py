import subprocess
import sys

import pytest

from molnova import schrodinger_retry as retry
from molnova.config import gbsa_license_retry_settings


def test_real_command_retries_license_failure_then_succeeds(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)
    script = """from pathlib import Path
import sys
p=Path('attempts')
n=int(p.read_text())+1 if p.exists() else 1
p.write_text(str(n))
if n==1:
    print('insufficient licenses for feature PSP_PLOP')
    sys.exit(1)
Path('prime-out.maegz').write_text('result')
"""
    retry.run_prime([sys.executable, "-c", script], tmp_path, wait_seconds=300, retries=3)
    assert waits == [300]
    assert (tmp_path / "attempts").read_text() == "2"


def test_retry_limit_and_current_log_detection(tmp_path, monkeypatch):
    waits = []
    calls = []
    monkeypatch.setattr(retry.time, "sleep", waits.append)

    def attempt(*args):
        calls.append(1)
        log = tmp_path / "prime.log"
        log.write_text('Unable to check out 8 PSP_PLOP license(s)\n')
        return 0, ""

    monkeypatch.setattr(retry, "_run_attempt", attempt)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        retry.run_prime(["prime"], tmp_path, wait_seconds=60, retries=2)
    assert waits == [60, 60]
    assert len(calls) == 3


def test_old_license_log_does_not_retry_unrelated_error(tmp_path, monkeypatch):
    (tmp_path / "prime.log").write_text("insufficient licenses")
    monkeypatch.setattr(retry, "_run_attempt", lambda *args: (1, "invalid input"))
    monkeypatch.setattr(retry.time, "sleep", lambda _: pytest.fail("unrelated error retried"))
    with pytest.raises(subprocess.CalledProcessError):
        retry.run_prime(["prime"], tmp_path)


def test_success_with_license_warning_is_not_retried(tmp_path, monkeypatch):
    def attempt(*args):
        (tmp_path / "prime-out.maegz").write_text("result")
        return 0, "license checkout failed earlier but recovered"
    monkeypatch.setattr(retry, "_run_attempt", attempt)
    monkeypatch.setattr(retry.time, "sleep", lambda _: pytest.fail("success retried"))
    retry.run_prime(["prime"], tmp_path)


@pytest.mark.parametrize("key,value", [("gbsa-license-retries", -1), ("gbsa-license-retry-seconds", 0), ("gbsa-license-retries", True)])
def test_invalid_license_retry_config(key, value):
    with pytest.raises(ValueError):
        gbsa_license_retry_settings({key: value})
