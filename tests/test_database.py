import sqlite3

import pytest

from molnova import _core
from molnova import database


def test_schema_contains_scores_and_state(tmp_path):
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(compound)")}
    assert {
        "docking_score",
        "gbsa_score",
        "fep_score",
        "synthetic_feasibility",
        "state",
    } <= cols


@pytest.mark.parametrize("state", ["failed", "gbsa_running"])
def test_schema_upgrade_preserves_existing_state(tmp_path, state):
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        conn.execute(
            """
            INSERT INTO compound (
                name, smiles, iteration, docking_score, state,
                failed_stage, failure_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("CMP", "C", 1, -8.0, state, "gbsa", "retry diagnostic"),
        )
        _core.ensure_schema_columns(conn)
        result = conn.execute(
            "SELECT state, failed_stage, failure_message "
            "FROM compound WHERE name='CMP'"
        ).fetchone()

    assert result == (state, "gbsa", "retry diagnostic")


def test_initialize_refuses_to_discard_compounds_without_references(tmp_path):
    db = tmp_path / "test.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.execute(
            "INSERT INTO compound(name, smiles, iteration) VALUES (?, ?, ?)",
            ("GEN_1", "C", 1),
        )

    with pytest.raises(RuntimeError, match="refusing to discard existing data"):
        _core.initialize_project_database(
            db_path=db,
            reference_poses=tmp_path / "unused.maegz",
            schrodinger=tmp_path / "schrodinger",
        )

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name, iteration FROM compound"
        ).fetchall()

    assert rows == [("GEN_1", 1)]


def test_legacy_state_is_inferred_once(tmp_path):
    db = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE compound (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                smiles TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                docking_score REAL,
                gbsa_score REAL,
                fep_score REAL,
                docking_status TEXT,
                ligprep_status TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO compound (
                name, smiles, iteration, docking_score,
                docking_status, ligprep_status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("REF", "C", 0, None, None, None),
                ("DOCKED", "CC", 1, -8.0, None, None),
                ("RUNNING", "CCC", 1, None, "running", None),
                ("LIGPREPPED", "CCCC", 1, None, None, "done"),
            ],
        )
        _core.ensure_schema_columns(conn)
        states = dict(conn.execute("SELECT name, state FROM compound"))

    assert states == {
        "REF": "reference",
        "DOCKED": "docked",
        "RUNNING": "glide_running",
        "LIGPREPPED": "ligprepped",
    }


def test_claim_compounds_only_transitions_eligible_ids(tmp_path):
    db = tmp_path / "claims.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        conn.executemany(
            "INSERT INTO compound(name, smiles, iteration, state) "
            "VALUES (?, ?, ?, ?)",
            [("A", "C", 1, "generated"), ("B", "CC", 1, "generated")],
        )
        ids = [row[0] for row in conn.execute("SELECT id FROM compound")]

    claimed = database.claim_compounds(db, ids, "generated", "ligprep_running")
    claimed_again = database.claim_compounds(
        db, ids, "generated", "ligprep_running"
    )

    assert set(claimed) == set(ids)
    assert claimed_again == []
    with sqlite3.connect(db) as conn:
        states = conn.execute(
            "SELECT state FROM compound ORDER BY id"
        ).fetchall()
    assert states == [("ligprep_running",), ("ligprep_running",)]


def test_database_helpers_keep_project_paths_isolated(tmp_path):
    project_a = tmp_path / "a.sqlite"
    project_b = tmp_path / "b.sqlite"
    for db, name, iteration in (
        (project_a, "A", 1),
        (project_b, "B", 2),
    ):
        with sqlite3.connect(db) as conn:
            _core.create_sqlite_schema(conn)
            _core.ensure_schema_columns(conn)
            conn.execute(
                "INSERT INTO compound(name, smiles, iteration, state) "
                "VALUES (?, ?, ?, 'generated')",
                (name, "C", iteration),
            )

    database.set_state(project_a, [1], "ligprep_running")
    database.set_state(project_b, [1], "failed")

    with sqlite3.connect(project_a) as conn:
        state_a = conn.execute(
            "SELECT state FROM compound WHERE id=1"
        ).fetchone()[0]
    with sqlite3.connect(project_b) as conn:
        state_b = conn.execute(
            "SELECT state FROM compound WHERE id=1"
        ).fetchone()[0]

    assert state_a == "ligprep_running"
    assert state_b == "failed"
    assert database.find_iteration("ligprep", project_a) == 1
    assert database.find_iteration("ligprep", project_b) is None


def test_claim_compounds_handles_large_batches(tmp_path):
    db = tmp_path / "large-claims.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        conn.executemany(
            "INSERT INTO compound(name, smiles, iteration, state) "
            "VALUES (?, ?, ?, 'generated')",
            [(f"CMP_{index}", "C", 1) for index in range(1000)],
        )
        ids = [row[0] for row in conn.execute("SELECT id FROM compound")]

    claimed = database.claim_compounds(db, ids, "generated", "ligprep_running")

    assert len(claimed) == 1000


def test_running_claim_can_be_recovered_after_stage_lock_releases(tmp_path):
    db = tmp_path / "recovery.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        conn.execute(
            "INSERT INTO compound(name, smiles, iteration, state) "
            "VALUES ('CMP', 'C', 1, 'generated')"
        )
        compound_id = conn.execute(
            "SELECT id FROM compound"
        ).fetchone()[0]

    with database.stage_lock(db, "ligprep"):
        claimed = database.claim_compounds(
            db,
            [compound_id],
            expected_state=("generated", "ligprep_running"),
            claimed_state="ligprep_running",
        )
        assert claimed == [compound_id]

    with database.stage_lock(db, "ligprep"):
        recovered = database.claim_compounds(
            db,
            [compound_id],
            expected_state=("generated", "ligprep_running"),
            claimed_state="ligprep_running",
        )

    assert recovered == [compound_id]


def test_stage_lock_rejects_overlapping_invocation(tmp_path):
    db = tmp_path / "locked.sqlite"

    with database.stage_lock(db, "mmgbsa"):
        with pytest.raises(RuntimeError, match="already running"):
            with database.stage_lock(db, "mmgbsa"):
                pass
