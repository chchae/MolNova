import gzip
import json
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from molnova import _core
from molnova.driver import build_stages, worker_loop
from molnova.stages import synthetic_feasibility
from molnova.chemistry.sa_score import calculate_sa_score


def _project(tmp_path):
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        conn.executemany(
            "INSERT INTO compound(name, smiles, iteration, state) "
            "VALUES (?, ?, 1, 'generated')",
            [("GEN_1", "C"), ("GEN_2", "CC")],
        )
    config = tmp_path / "aizynth.yml"
    config.write_text("search: {}\n", encoding="utf-8")
    return SimpleNamespace(
        db_path=db,
        output=tmp_path / "output",
        synthetic_feasibility_enabled=True,
        aizynth_config=config,
        aizynth_cli="aizynthcli",
    )


def test_worker_persists_solved_route_feasibility(monkeypatch, tmp_path):
    args = _project(tmp_path)
    monkeypatch.setattr(synthetic_feasibility.shutil, "which", lambda _: "/bin/aizynthcli")

    def run(command, **kwargs):
        result_file = command[command.index("--output") + 1]
        rows = [
            {"target": "C", "number_of_solved_routes": 2},
            {"target": "CC", "number_of_solved_routes": 0},
        ]
        with gzip.open(result_file, "wt", encoding="utf-8") as stream:
            json.dump({"schema": {}, "data": rows}, stream)

    monkeypatch.setattr(synthetic_feasibility.subprocess, "run", run)

    assert synthetic_feasibility.process_iteration(args, 1) == 2
    with sqlite3.connect(args.db_path) as conn:
        rows = conn.execute(
            "SELECT name, synthetic_feasibility, state "
            "FROM compound ORDER BY id"
        ).fetchall()

    assert rows == [
        ("GEN_1", 1, "generated"),
        ("GEN_2", 0, "generated"),
    ]
    with sqlite3.connect(args.db_path) as conn:
        scores = conn.execute("SELECT smiles, sa_score FROM compound ORDER BY id").fetchall()
    assert len(scores) == 2
    for smiles, score in scores:
        assert score == pytest.approx(calculate_sa_score(smiles))
    # A second invocation must not rerun the route search or score calculation.
    monkeypatch.setattr(synthetic_feasibility, "calculate_sa_score", lambda _: pytest.fail("rescored"))
    monkeypatch.setattr(synthetic_feasibility.subprocess, "run", lambda *a, **k: pytest.fail("searched again"))
    assert synthetic_feasibility.process_iteration(args, 1) == 0


def test_worker_restores_claims_after_cli_failure(monkeypatch, tmp_path):
    args = _project(tmp_path)
    monkeypatch.setattr(synthetic_feasibility.shutil, "which", lambda _: "/bin/aizynthcli")

    def fail(*args, **kwargs):
        raise RuntimeError("search failed")

    monkeypatch.setattr(synthetic_feasibility.subprocess, "run", fail)

    try:
        synthetic_feasibility.process_iteration(args, 1)
    except RuntimeError as exc:
        assert str(exc) == "search failed"
    else:
        raise AssertionError("Expected the CLI failure to propagate")

    with sqlite3.connect(args.db_path) as conn:
        states = conn.execute(
            "SELECT state, failed_stage FROM compound ORDER BY id"
        ).fetchall()
    assert states == [
        ("generated", "synthetic_feasibility"),
        ("generated", "synthetic_feasibility"),
    ]
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT COUNT(sa_score) FROM compound").fetchone()[0] == 2


def test_missing_sa_scores_backfilled_without_changing_scientific_results(monkeypatch, tmp_path):
    args = _project(tmp_path)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute(
            "UPDATE compound SET synthetic_feasibility=1, state='gbsa_done', "
            "docking_score=-8, gbsa_score=-40"
        )
        conn.execute("UPDATE compound SET sa_score=3.25 WHERE name='GEN_1'")
    args.aizynth_config = None  # Backfill must not require route-search dependencies.
    scored = []

    def score(smiles):
        scored.append(smiles)
        return calculate_sa_score(smiles)

    monkeypatch.setattr(synthetic_feasibility, "calculate_sa_score", score)
    monkeypatch.setattr(synthetic_feasibility.subprocess, "run", lambda *a, **k: pytest.fail("route search rerun"))
    assert synthetic_feasibility.choose_iteration(args, None) == 1
    assert synthetic_feasibility.process_iteration(args, 1) == 0
    assert scored == ["CC"]
    assert synthetic_feasibility.choose_iteration(args, None) is None
    with sqlite3.connect(args.db_path) as conn:
        rows = conn.execute(
            "SELECT sa_score, synthetic_feasibility, state, docking_score, gbsa_score "
            "FROM compound ORDER BY id"
        ).fetchall()
    assert rows[0] == (3.25, 1, "gbsa_done", -8, -40)
    assert rows[1] == (pytest.approx(calculate_sa_score("CC")), 1, "gbsa_done", -8, -40)


@pytest.mark.parametrize("smiles", ["not a smiles", ""])
def test_sa_score_rejects_invalid_or_empty_smiles(smiles):
    with pytest.raises(ValueError, match="invalid or empty SMILES"):
        calculate_sa_score(smiles)


def test_sa_score_uses_bundled_rdkit_scorer():
    assert calculate_sa_score("CCO") == pytest.approx(1.9802570386349831)


def test_scoring_failure_does_not_write_partial_batch_or_claim_compounds(tmp_path):
    args = _project(tmp_path)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET smiles='invalid' WHERE name='GEN_2'")
    with pytest.raises(ValueError):
        synthetic_feasibility.process_iteration(args, 1)
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT sa_score, state FROM compound ORDER BY id").fetchall() == [
            (None, "generated"), (None, "generated")
        ]


def test_driver_places_synthetic_stage_after_generation():
    stages = build_stages(synthetic_feasibility_enabled=True, fep_enabled=False)
    names = [stage for stage, _ in stages]

    assert names == ["generate", "synthetic", "ligprep", "glide", "mmgbsa"]


def test_once_evaluates_compounds_after_generate_finishes(monkeypatch, tmp_path):
    args = _project(tmp_path)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("DELETE FROM compound")
    waiting = threading.Event()
    observed = []

    class WakeEvent(threading.Event):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)

    wake = WakeEvent()
    stop = threading.Event()

    def process(stage, *unused):
        with sqlite3.connect(args.db_path) as conn:
            if stage == "generate":
                conn.execute(
                    "INSERT INTO compound(name, smiles, iteration, state) "
                    "VALUES ('NEW', 'C', 1, 'generated')"
                )
            else:
                observed.extend(conn.execute("SELECT name FROM compound").fetchall())
        return 0

    monkeypatch.setattr("molnova.driver.stream_process", process)
    synthetic = threading.Thread(
        target=worker_loop,
        args=("synthetic", "synthetic", tmp_path / "project.toml", tmp_path, 30, stop, True),
        kwargs={"wake_event": wake},
    )
    synthetic.start()
    try:
        assert waiting.wait(2)
        assert observed == []
        worker_loop(
            "generate", "generate", tmp_path / "project.toml", tmp_path,
            30, stop, True, after_run_event=wake,
        )
        synthetic.join(2)
        assert not synthetic.is_alive()
        assert observed == [("NEW",)]
    finally:
        stop.set()
        wake.set()
        synthetic.join(2)


def test_generation_completion_wakes_idle_synthetic_worker(monkeypatch, tmp_path):
    idle = threading.Event()
    processed = threading.Event()
    stop = threading.Event()
    calls = []

    class WakeEvent(threading.Event):
        def wait(self, timeout=None):
            idle.set()
            return super().wait(timeout)

    wake = WakeEvent()

    def process(stage, *unused):
        if stage == "synthetic":
            calls.append(stage)
            if len(calls) == 2:
                processed.set()
                stop.set()
        return 0

    monkeypatch.setattr("molnova.driver.stream_process", process)
    synthetic = threading.Thread(
        target=worker_loop,
        args=("synthetic", "synthetic", tmp_path / "project.toml", tmp_path, 60, stop, False),
        kwargs={"wake_event": wake},
    )
    synthetic.start()
    try:
        assert idle.wait(2)
        worker_loop(
            "generate", "generate", tmp_path / "project.toml", tmp_path,
            60, stop, True, after_run_event=wake,
        )
        assert processed.wait(2), "Synthetic worker waited for the 60-second polling timeout"
        synthetic.join(2)
        assert calls == ["synthetic", "synthetic"]
    finally:
        stop.set()
        wake.set()
        synthetic.join(2)


@pytest.mark.parametrize("flag, enabled", [("", True), ("true", True), ("false", False)])
def test_aizynth_config_paths_resolve_from_project_toml(tmp_path, flag, enabled):
    model_config = tmp_path / "models" / "aizynth.yml"
    model_config.parent.mkdir()
    model_config.write_text("search: {}\n", encoding="utf-8")
    project = tmp_path / "project.toml"
    project.write_text(
        "\n".join(
            [
                'project = "test"',
                'libinvent-prior = "prior.prior"',
                'dock-grid = "grid.zip"',
                'reference-pose = "references.maegz"',
                'out-dir = "output"',
                "sample-size = 10",
                "target-count = 5",
                "max-iteration = 2",
                'host = "localhost"',
                f"synthetic-feasibility-enabled = {flag}" if flag else "",
                'aizynth-config = "models/aizynth.yml"',
                'aizynth-cli = "/opt/aizynth/bin/aizynthcli"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = _core.load_project_toml(project)

    assert config["synthetic_feasibility_enabled"] is enabled
    assert config["aizynth_config"] == model_config
    assert config["aizynth_cli"] == "/opt/aizynth/bin/aizynthcli"


def test_import_preserves_completed_and_failed_scientific_results(tmp_path):
    args = _project(tmp_path)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='gbsa_done', docking_score=-8, gbsa_score=-40 WHERE id=1")
        conn.execute("UPDATE compound SET state='failed', failed_stage='glide', failure_message='no poses' WHERE id=2")
        before = conn.execute("SELECT state, docking_score, gbsa_score, failed_stage, failure_message FROM compound ORDER BY id").fetchall()
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"data": [
        {"target": "C", "number_of_solved_routes": 2},
        {"target": "CC", "number_of_solved_routes": 0},
    ]}))
    assert synthetic_feasibility.import_results(args, 1, results) == 2
    assert synthetic_feasibility.import_results(args, 1, results) == 0
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT state, docking_score, gbsa_score, failed_stage, failure_message FROM compound ORDER BY id").fetchall() == before
        assert conn.execute("SELECT synthetic_feasibility FROM compound ORDER BY id").fetchall() == [(1,), (0,)]
        assert conn.execute("SELECT COUNT(sa_score) FROM compound").fetchone()[0] == 2


@pytest.mark.parametrize("case", ["unknown", "conflict", "duplicate"])
def test_import_rejects_invalid_batch_without_partial_updates(tmp_path, case):
    args = _project(tmp_path)
    rows = [{"target": "C", "number_of_solved_routes": 1},
            {"target": "CC", "number_of_solved_routes": 0}]
    if case == "unknown":
        rows[1]["target"] = "CCC"
    elif case == "duplicate":
        rows[1]["target"] = "C"
    else:
        with sqlite3.connect(args.db_path) as conn:
            conn.execute("UPDATE compound SET synthetic_feasibility=1 WHERE id=2")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"data": rows}))
    with pytest.raises(RuntimeError):
        synthetic_feasibility.import_results(args, 1, results)
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT sa_score, synthetic_feasibility FROM compound WHERE id=1").fetchone() == (None, None)


def test_import_cli_forwards_result_path(monkeypatch):
    from molnova import cli
    calls = []
    monkeypatch.setattr(synthetic_feasibility, "main", lambda args: calls.append(args))
    cli.main(["synthetic-feasibility", "egfr.toml", "--iteration", "1", "--import-results", "results.json.gz"])
    assert calls == [["egfr.toml", "--iteration", "1", "--import-results", "results.json.gz"]]


def test_remote_aizynth_roundtrip_persists_local_sqlite(tmp_path, monkeypatch):
    from molnova import aizynth_remote
    args = _project(tmp_path)
    env = tmp_path / "remote env"
    (env / "bin").mkdir(parents=True)
    executable = env / "bin/aizynthcli"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import sys, gzip, json, pathlib\n"
        "smiles=pathlib.Path(sys.argv[sys.argv.index('--smiles')+1]).read_text().splitlines()\n"
        "output=sys.argv[sys.argv.index('--output')+1]\n"
        "with gzip.open(output, 'wt') as stream:\n"
        "    json.dump({'data':[{'target':s,'number_of_solved_routes':int(s=='C')} for s in smiles]}, stream)\n"
    )
    executable.chmod(0o755)
    args.remote_aizynth = {"host": "tensor", "env": str(env),
                           "work_dir": str(tmp_path / "remote jobs"), "config": "/remote/config.yml"}
    args.aizynth_config = None
    monkeypatch.setattr(synthetic_feasibility.shutil, "which", lambda _: pytest.fail("local CLI used"))

    def ssh(settings, command, **kwargs):
        return subprocess.run(["bash", "-c", command], check=True, **kwargs)

    monkeypatch.setattr(aizynth_remote, "_ssh", ssh)
    assert synthetic_feasibility.process_iteration(args, 1) == 2
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT synthetic_feasibility, state FROM compound ORDER BY id").fetchall() == [(1, "generated"), (0, "generated")]
        assert conn.execute("SELECT COUNT(sa_score) FROM compound").fetchone()[0] == 2


@pytest.mark.parametrize("failure", ["upload", "execution", "download", "empty"])
def test_remote_failure_restores_claims_and_rejects_stale_results(tmp_path, monkeypatch, failure):
    from molnova import aizynth_remote
    args = _project(tmp_path)
    args.remote_aizynth = {"host": "tensor", "env": "/remote/env",
                           "work_dir": "/remote/jobs", "config": "/remote/config.yml"}
    output = args.output / "iter1/synthetic_feasibility/aizynthfinder_results.json.gz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"stale")

    def ssh(settings, command, **kwargs):
        if ((failure == "upload" and "stdin" in kwargs)
                or (failure == "execution" and "export PATH=" in command)
                or (failure == "download" and "cat --" in command)):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(aizynth_remote, "_ssh", ssh)
    with pytest.raises(RuntimeError):
        synthetic_feasibility.process_iteration(args, 1)
    assert output.read_bytes() == b"stale"
    assert not output.with_name(output.name + ".download").exists()
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT state, synthetic_feasibility FROM compound").fetchall() == [("generated", None), ("generated", None)]
