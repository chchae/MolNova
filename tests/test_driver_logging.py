import sys
import threading

import pytest

from molnova.driver import stream_process
from molnova.stages._logging import WORK_STARTED, work_started


@pytest.mark.parametrize("active,failed", [(False, False), (True, False), (False, True), (True, True)])
def test_supervisor_logs_only_work_and_preserves_errors(tmp_path, capsys, active, failed):
    script = (
        "import os, sys\n"
        "assert os.environ['MOLNOVA_SUPERVISED'] == '1'\n"
        "assert os.environ['PYTHONUNBUFFERED'] == '1'\n"
        "print('Using existing SQLite project database')\n"
        "print('MCS matched references: 23/24')\n"
    )
    if active:
        script += f"print({WORK_STARTED!r})\nprint('Starting LigPrep for 1000 compounds')\n"
    else:
        script += "print('No iteration requires LigPrep.')\n"
    if failed:
        script += "raise RuntimeError('calculation or initialization failed')\n"
    rc = stream_process("ligprep", [sys.executable, "-c", script], tmp_path, threading.Event())
    output = capsys.readouterr().out
    assert (rc != 0) == failed
    assert WORK_STARTED not in output
    if active:
        assert "Starting LigPrep for 1000 compounds" in output
        assert "MCS matched references" not in output
    elif not failed:
        assert output == ""
    else:
        assert "Using existing SQLite project database" in output
    if failed:
        assert "Traceback" in output
        assert "calculation or initialization failed" in output


def test_standalone_work_message_has_no_protocol_marker(monkeypatch, capsys):
    monkeypatch.delenv("MOLNOVA_SUPERVISED", raising=False)
    work_started("Starting MM-GBSA")
    assert capsys.readouterr().out == "Starting MM-GBSA\n"
