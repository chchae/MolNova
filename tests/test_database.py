import sqlite3
from molnova import _core


def test_schema_contains_scores_and_state(tmp_path):
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(compound)")}
    assert {"docking_score", "gbsa_score", "fep_score", "state"} <= cols
