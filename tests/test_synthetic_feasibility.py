import gzip
import json
import sqlite3
from types import SimpleNamespace

from molnova import _core
from molnova.driver import build_stages
from molnova.stages import synthetic_feasibility


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


def test_driver_places_synthetic_stage_after_generation():
    stages = build_stages(synthetic_feasibility_enabled=True, fep_enabled=False)
    names = [stage for stage, _ in stages]

    assert names == ["generate", "synthetic", "ligprep", "glide", "mmgbsa"]


def test_aizynth_config_paths_resolve_from_project_toml(tmp_path):
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
                "synthetic-feasibility-enabled = true",
                'aizynth-config = "models/aizynth.yml"',
                'aizynth-cli = "/opt/aizynth/bin/aizynthcli"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = _core.load_project_toml(project)

    assert config["synthetic_feasibility_enabled"] is True
    assert config["aizynth_config"] == model_config
    assert config["aizynth_cli"] == "/opt/aizynth/bin/aizynthcli"