import json
import subprocess
from types import SimpleNamespace
import sqlite3

import pytest

from molnova import _core, prime_async as prime
from molnova.stages import mmgbsa


def test_submit_then_poll_without_resubmission_or_early_result_read(tmp_path, monkeypatch):
    launches = []
    def launch(command, cwd):
        launches.append(command)
        assert "-WAIT" not in command
        return 0, "JobId: job-1\n"
    monkeypatch.setattr(prime, "_run_attempt", launch)
    command = ["/suite/prime_mmgbsa", "input.maegz", "-HOST", "compute:4", "-NJOBS", "4"]
    assert prime.submit_or_poll(command, tmp_path, [1, 2]) is False
    assert prime.load_job(tmp_path)["compound_ids"] == [1, 2]
    status = "RUNNING"
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        if command[1] == "info":
            # jsc also returns child jobs, one JSON object per line.
            return SimpleNamespace(stdout=json.dumps({"jobId": "child", "status": "DONE"}) +
                                   "\n" + json.dumps({"jobId": "job-1", "status": status}))
        (tmp_path / "input-out.maegz").write_text("complete output")
        return SimpleNamespace(stdout="")
    monkeypatch.setattr(prime.subprocess, "run", query)
    assert prime.submit_or_poll(command, tmp_path, [999]) is False
    assert not any(c[1] == "download" for c in calls)
    status = "DONE"
    assert prime.submit_or_poll(command, tmp_path, [999]) is True
    assert len(launches) == 1
    assert calls[-1] == ["/suite/jsc", "download", "--cwd", "job-1"]


def test_failed_job_is_retryable_and_license_retry_is_deferred(tmp_path, monkeypatch):
    monkeypatch.setattr(prime, "_run_attempt", lambda *a: (0, "JobId: job-1"))
    command = ["/suite/prime_mmgbsa", "input.maegz"]
    prime.submit_or_poll(command, tmp_path, [1])
    monkeypatch.setattr(prime.subprocess, "run", lambda cmd, **kw:
                        SimpleNamespace(stdout=json.dumps({"jobId": "job-1", "status": "FAILED"})))
    (tmp_path / "input.log").write_text("insufficient licenses")
    assert prime.submit_or_poll(command, tmp_path, [1], wait_seconds=300) is False
    assert prime.load_job(tmp_path)["retry_at"] > prime.time.time()
    record = prime.load_job(tmp_path)
    record.update(job_id="job-1", retry_at=0)
    prime._save(tmp_path, record)
    (tmp_path / "input.log").write_text("atom typing failed")
    with pytest.raises(RuntimeError, match="atom typing failed"):
        prime.submit_or_poll(command, tmp_path, [1])
    assert prime.load_job(tmp_path) is None


def test_worker_submits_multiple_iterations_and_preserves_running_states(tmp_path, monkeypatch):
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany("INSERT INTO compound(name,smiles,iteration,docking_score,state) "
                         "VALUES (?,'C',?,-9,'docked')", [("A", 1), ("B", 2)])
    args = SimpleNamespace(db_path=db, gbsa_input_count=2, output=tmp_path)
    calls = []
    def submit(**kw):
        calls.append(kw["iteration"])
        return None
    monkeypatch.setattr(_core, "run_iteration_mmgbsa", submit)
    mmgbsa._run_stage(args, None)
    assert calls == [1, 2]
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT state,failed_stage FROM compound").fetchall() == [
            ("gbsa_running", None), ("gbsa_running", None)]


def test_recovery_uses_submitted_ids_even_after_top_n_changes(tmp_path, monkeypatch):
    directory = tmp_path / "iter1/mmgbsa"
    directory.mkdir(parents=True)
    prime._save(directory, {"command": ["/suite/prime_mmgbsa"], "compound_ids": [5]})
    args = SimpleNamespace()
    monkeypatch.setattr(prime, "submit_or_poll", lambda *a, **kw: True)
    calls = []
    monkeypatch.setattr(_core, "finish_iteration_mmgbsa",
                        lambda iteration, args, directory, ids: calls.append(ids) or 1)
    assert _core.run_iteration_mmgbsa(1, args, tmp_path / "iter1", tmp_path / "glide") == 1
    assert calls == [[5]]
    assert not prime.job_file(directory).exists()


def test_status_lookup_failure_keeps_job_and_running_compound(tmp_path, monkeypatch):
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name,smiles,iteration,docking_score,state) "
                     "VALUES ('A','C',1,-9,'gbsa_running')")
    directory = tmp_path / "iter1/mmgbsa"
    directory.mkdir(parents=True)
    prime._save(directory, {"command": ["/suite/prime_mmgbsa"], "compound_ids": [1],
                            "job_id": "job-1"})
    def offline(*a, **kw):
        raise subprocess.CalledProcessError(1, "jsc")
    monkeypatch.setattr(prime.subprocess, "run", offline)
    args = SimpleNamespace(db_path=db, gbsa_input_count=2, output=tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        mmgbsa._run_stage(args, 1)
    assert prime.load_job(directory)["job_id"] == "job-1"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT state FROM compound").fetchone() == ("gbsa_running",)


def test_interrupted_submission_does_not_launch_duplicate(tmp_path, monkeypatch):
    prime._save(tmp_path, {"command": ["/suite/prime_mmgbsa"], "compound_ids": [1],
                          "job_id": None, "launching": True})
    monkeypatch.setattr(prime, "_run_attempt", lambda *a: pytest.fail("duplicate submission"))
    with pytest.raises(RuntimeError, match="interrupted"):
        prime.submit_or_poll([], tmp_path, [1])


@pytest.mark.parametrize("status", ["FAILED", "DONE"])
def test_terminal_single_atomtyping_failure_is_classified_even_with_output(tmp_path, monkeypatch, status):
    command = ["/suite/prime_mmgbsa", "input.maegz"]
    monkeypatch.setattr(prime, "_run_attempt", lambda *a: (0, "JobId: job-1"))
    prime.submit_or_poll(command, tmp_path, [11379])
    def query(cmd, **kwargs):
        if cmd[1] == "download":
            (tmp_path / "input.log").write_text(
                "JobId: job-1\nCMPID_11379 ERROR running plop library\nProblem in atomtyping structure.")
            (tmp_path / "input-out.maegz").write_text("failed entry output")
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout=json.dumps({"jobId": "job-1", "status": status}))
    monkeypatch.setattr(prime.subprocess, "run", query)
    with pytest.raises(prime.PrimeAtomTypingError) as error:
        prime.submit_or_poll(command, tmp_path, [999])
    assert error.value.compound_id == 11379
    assert prime.load_job(tmp_path) is None


def test_inline_atomtyping_failure_is_classified(tmp_path, monkeypatch):
    monkeypatch.setattr(prime, "_run_attempt", lambda *a: (
        1, "CMPID_11379 ERROR running plop library\nFailure running atom typer"))
    with pytest.raises(prime.PrimeAtomTypingError):
        prime.submit_or_poll(["/suite/prime_mmgbsa"], tmp_path, [11379])
    assert prime.load_job(tmp_path) is None


@pytest.mark.parametrize("ids,text", [
    ([11379, 11380], "CMPID_11379 Cannot atom type structure"),
    ([11379], "CMPID_113790 Cannot atom type structure"),
    ([11379], "CMPID_11379 ERROR running plop library"),
])
def test_ambiguous_or_generic_failure_is_not_classified(ids, text):
    prime._check_single_compound_atomtyping(text, ids)


def test_atomtyping_failure_stops_retries_and_allows_next_iteration(tmp_path, monkeypatch):
    from molnova import database
    from molnova.states import CompoundState as State
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany("INSERT INTO compound(name,smiles,iteration,docking_score,gbsa_score,state) "
                         "VALUES (?,'C',?,?,?,?)", [
                             ("bad", 1, -9, None, State.DOCKED),
                             ("done", 1, -10, -40, State.GBSA_DONE),
                             ("next", 2, -8, None, State.DOCKED)])
    args = SimpleNamespace(db_path=db, gbsa_input_count=2, output=tmp_path)
    calls = []
    def calculate(**kw):
        calls.append(kw["iteration"])
        if kw["iteration"] == 1:
            raise prime.PrimeAtomTypingError(1)
        return None
    monkeypatch.setattr(_core, "run_iteration_mmgbsa", calculate)
    mmgbsa._run_stage(args, None)
    mmgbsa._run_stage(args, None)
    assert calls == [1, 2, 2]
    assert database.mmgbsa_candidates(db, 1, 2) == []
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT state,docking_score,gbsa_score,failed_stage,failure_message "
                            "FROM compound ORDER BY id").fetchall()
    assert rows[0][:4] == (State.FAILED, -9, None, "gbsa")
    assert "atom typing failed" in rows[0][4]
    assert rows[1] == (State.GBSA_DONE, -10, -40, None, None)
    assert rows[2][:4] == (State.GBSA_RUNNING, -8, None, None)


def test_old_atomtyping_log_does_not_classify_new_job(tmp_path, monkeypatch):
    command = ["/suite/prime_mmgbsa", "input.maegz"]
    monkeypatch.setattr(prime, "_run_attempt", lambda *a: (0, "JobId: job-new"))
    prime.submit_or_poll(command, tmp_path, [11379])
    (tmp_path / "old.log").write_text(
        "JobId: job-old\nCMPID_11379 Cannot atom type structure")
    def query(cmd, **kwargs):
        if cmd[1] == "download":
            (tmp_path / "input.log").write_text("JobId: job-new\ncalculation interrupted")
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout=json.dumps({"jobId": "job-new", "status": "FAILED"}))
    monkeypatch.setattr(prime.subprocess, "run", query)
    with pytest.raises(RuntimeError) as error:
        prime.submit_or_poll(command, tmp_path, [11379])
    assert not isinstance(error.value, prime.PrimeAtomTypingError)
