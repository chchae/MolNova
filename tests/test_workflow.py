import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core
from molnova.driver import supervise_iteration, StopFlag
from molnova.stages.fep import import_scores


def _fep_args(tmp_path, fep_input_count=2):
    db = tmp_path / "fep.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        conn.executemany(
            """
            INSERT INTO compound(name, smiles, iteration, gbsa_score, state)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("REF", "C", 0, None, "reference"),
                ("TOP1", "CC", 1, -10.0, "gbsa_done"),
                ("TOP2", "CCC", 1, -9.0, "gbsa_done"),
                ("OUTSIDE", "CCCC", 1, -8.0, "gbsa_done"),
            ],
        )
        ids = dict(conn.execute("SELECT name, id FROM compound"))
    return SimpleNamespace(db_path=db, fep_input_count=fep_input_count), ids


def _write_scores(path: Path, rows):
    path.write_text(
        "id\tfep_score\n"
        + "".join(f"{compound_id}\t{score}\n" for compound_id, score in rows),
        encoding="utf-8",
    )


def test_fep_import_updates_selected_candidate(tmp_path):
    args, ids = _fep_args(tmp_path)
    scores = tmp_path / "scores.tsv"
    _write_scores(scores, [(ids["TOP2"], -7.5)])

    assert import_scores(args, scores, iteration=1) == 1
    with sqlite3.connect(args.db_path) as conn:
        row = conn.execute(
            "SELECT fep_score, state FROM compound WHERE id=?",
            (ids["TOP2"],),
        ).fetchone()
    assert row == (-7.5, "fep_done")


@pytest.mark.parametrize("name", ["REF", "OUTSIDE"])
def test_fep_import_rejects_ids_outside_candidate_set(tmp_path, name):
    args, ids = _fep_args(tmp_path, fep_input_count=1)
    scores = tmp_path / "scores.tsv"
    _write_scores(scores, [(ids[name], -7.5)])

    with pytest.raises(ValueError, match="outside the selected GBSA candidate set"):
        import_scores(args, scores, iteration=1)


def test_one_shot_worker_does_not_retry_failure(monkeypatch):
    calls = []

    def failed_process(*args):
        calls.append(args)
        return 1

    monkeypatch.setattr("molnova.driver.stream_process", failed_process)
    assert not supervise_iteration(
        SimpleNamespace(), Path("project.toml"), 1,
        [("generate", "molnova.stages.generate")], 1, StopFlag(), True,
    )

    assert len(calls) == 1