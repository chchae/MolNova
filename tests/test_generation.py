import sqlite3
from types import SimpleNamespace

import pytest

from molnova import _core
from molnova.stages.generate import choose_iteration


def _project_args(tmp_path, max_iteration=3, target_count=2):
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
    return SimpleNamespace(
        db_path=db,
        max_iteration=max_iteration,
        target_count=target_count,
    )


def test_choose_iteration_starts_at_one(tmp_path):
    args = _project_args(tmp_path)

    assert choose_iteration(args, None) == 1


def test_choose_iteration_resumes_underfilled_latest_iteration(tmp_path):
    args = _project_args(tmp_path)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute(
            "INSERT INTO compound(name, smiles, iteration) VALUES (?, ?, ?)",
            ("GEN_1", "C", 1),
        )

    assert choose_iteration(args, None) == 1


def test_choose_iteration_advances_or_stops_at_limit(tmp_path):
    args = _project_args(tmp_path, max_iteration=2)
    with sqlite3.connect(args.db_path) as conn:
        conn.executemany(
            "INSERT INTO compound(name, smiles, iteration) VALUES (?, ?, ?)",
            [("GEN_1", "C", 1), ("GEN_2", "CC", 1)],
        )

    assert choose_iteration(args, None) == 2
    args.max_iteration = 1
    assert choose_iteration(args, None) is None


def test_choose_iteration_rejects_explicit_iteration_over_limit(tmp_path):
    args = _project_args(tmp_path, max_iteration=2)

    with pytest.raises(ValueError, match="between 1 and 2"):
        choose_iteration(args, 3)


def test_insert_generated_is_idempotent_for_existing_smiles(tmp_path):
    args = _project_args(tmp_path)
    generated = tmp_path / "generated.csv"
    generated.write_text("SMILES\nC\nCC\n", encoding="utf-8")

    first = _core.insert_generated(
        generated,
        iteration=1,
        target_count=2,
        db_path=args.db_path,
    )
    second = _core.insert_generated(
        generated,
        iteration=1,
        target_count=2,
        db_path=args.db_path,
    )

    assert first == 2
    assert second == 0
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM compound WHERE iteration=1"
        ).fetchone()[0] == 2