import sys
import threading

import pytest

from molnova.driver import stream_process
from molnova.stages._logging import WORK_STARTED, work_started


@pytest.mark.parametrize("active,failed", [(False, False), (True, False), (False, True), (True, True)])
def test_supervisor_summarizes_idle_work_and_preserves_errors(tmp_path, capsys, active, failed):
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
        assert output == "[ligprep ] skip: No iteration requires LigPrep.\n"
    else:
        assert "Using existing SQLite project database" in output
    if failed:
        assert "Traceback" in output
        assert "calculation or initialization failed" in output


def test_standalone_work_message_has_no_protocol_marker(monkeypatch, capsys):
    monkeypatch.delenv("MOLNOVA_SUPERVISED", raising=False)
    work_started("Starting MM-GBSA")
    assert capsys.readouterr().out == "Starting MM-GBSA\n"


@pytest.mark.parametrize("stage", ["generate", "synthetic", "ligprep", "glide", "mmgbsa", "fep"])
@pytest.mark.parametrize("silent", [False, True])
def test_each_idle_stage_emits_one_skip_line(tmp_path, capsys, stage, silent):
    script = "pass" if silent else "print('setup message'); print('No eligible work.'); print()"
    assert stream_process(stage, [sys.executable, "-c", script], tmp_path, threading.Event()) == 0
    assert capsys.readouterr().out == f"[{stage:<8}] skip: No eligible work.\n"


def test_shutdown_does_not_report_skip(tmp_path, capsys):
    stop = threading.Event()
    stop.set()
    stream_process("glide", [sys.executable, "-c", "pass"], tmp_path, stop)
    assert "skip:" not in capsys.readouterr().out


def test_mmgbsa_idle_poll_has_explicit_reason(tmp_path, monkeypatch, capsys):
    import sqlite3
    from types import SimpleNamespace
    from molnova import _core
    from molnova.stages import mmgbsa
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
    args = SimpleNamespace(db_path=db, output=tmp_path, gbsa_input_count=100)
    mmgbsa._run_stage(args, None)
    assert capsys.readouterr().out == "No current docking top-N compounds require MM-GBSA.\n"
