"""SQLite workflow-state API."""
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterable, Iterator
import fcntl
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from molnova import _core
from molnova.states import CompoundState as State


def connect(path: str | Path) -> sqlite3.Connection:
    return _core.open_sqlite(Path(path))


def lock_path(path: str | Path, suffix: str) -> Path:
    """Keep workflow locks beside the project SQLite file in its output folder."""
    db_path = Path(path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path.with_name(f"{db_path.name}.{suffix}.lock")


def migrate_project_database(legacy_path: str | Path, path: str | Path) -> bool:
    """Copy a legacy DB to the output directory, including committed WAL data."""
    legacy_path, path = Path(legacy_path), Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or not legacy_path.is_file():
        return False
    with lock_path(path, "migration").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            return False
        fd, name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".migration", dir=path.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            source = sqlite3.connect(legacy_path.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                target = sqlite3.connect(temporary)
                try:
                    source.backup(target)
                finally:
                    target.close()
            finally:
                source.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Copied existing project database to {path}; original retained at {legacy_path}.")
    return True


def initialize(
    project_toml: str | Path, schrodinger: str | Path | None = None
) -> SimpleNamespace:
    return _core.configure_project(project_toml, Path(schrodinger) if schrodinger else None)


@contextmanager
def stage_lock(path: str | Path, stage: str) -> Iterator[None]:
    """Prevent overlapping local invocations of one project stage."""
    lock_file = lock_path(path, f"{stage}.driver").open("a")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError(f"Stage {stage} is already running for {path}") from exc
    try:
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def set_state(
    path: str | Path,
    ids: Iterable[int],
    state: str,
    failed_stage: str | None = None,
    failure_message: str | None = None,
) -> None:
    return _core.set_compound_state(
        ids,
        state,
        failed_stage,
        failure_message,
        db_path=path,
    )


def claim_compounds(
    path: str | Path,
    ids: Iterable[int],
    expected_state: str | tuple[str, ...],
    claimed_state: str,
) -> list[int]:
    """Atomically transition eligible compounds and return claimed IDs."""
    ids = list(dict.fromkeys(ids))
    if not ids:
        return []
    expected_states = (
        (expected_state,) if isinstance(expected_state, str) else expected_state
    )
    if not expected_states:
        return []
    if expected_states == (claimed_state,):
        raise ValueError("Claimed state must differ from expected state")

    conn = _core.open_sqlite(Path(path))
    try:
        conn.execute("BEGIN IMMEDIATE")
        state_marks = ",".join("?" for _ in expected_states)
        claimed_ids = []
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            marks = ",".join("?" for _ in batch)
            eligible = conn.execute(
                f"SELECT id FROM compound WHERE state IN ({state_marks}) "
                f"AND id IN ({marks})",
                [*expected_states, *batch],
            ).fetchall()
            claimed_ids.extend(row[0] for row in eligible)
        if not claimed_ids:
            conn.commit()
            return []

        for start in range(0, len(claimed_ids), 500):
            batch = claimed_ids[start : start + 500]
            marks = ",".join("?" for _ in batch)
            conn.execute(
                f"""
                UPDATE compound
                SET state=?, failed_stage=NULL, failure_message=NULL,
                    modified_at=CURRENT_TIMESTAMP
                WHERE state IN ({state_marks}) AND id IN ({marks})
                """,
                [claimed_state, *expected_states, *batch],
            )
        conn.commit()
        return claimed_ids
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def find_iteration(stage: str, path: str | Path) -> int | None:
    return _core.find_iteration_for_state(stage, path)


def mmgbsa_candidates(path: str | Path, iteration: int, count: int) -> list[int]:
    """Rank all docking results first, then select unscored, eligible top-N IDs."""
    with connect(path) as conn:
        rows = conn.execute(
            """
            SELECT id FROM (
                SELECT id, state, gbsa_score, docking_score FROM compound
                WHERE iteration=? AND iteration>0 AND docking_score IS NOT NULL
                ORDER BY docking_score ASC, id ASC LIMIT ?
            )
            WHERE gbsa_score IS NULL AND state IN (?, ?)
            ORDER BY docking_score ASC, id ASC
            """,
            (iteration, count, State.DOCKED, State.GBSA_RUNNING),
        ).fetchall()
    return [row[0] for row in rows]


def missing_sa_scores(path: str | Path, iteration: int) -> list[tuple[int, str]]:
    with connect(path) as conn:
        return conn.execute(
            "SELECT id, smiles FROM compound "
            "WHERE iteration=? AND iteration>0 AND sa_score IS NULL ORDER BY id",
            (iteration,),
        ).fetchall()


def store_sa_scores(path: str | Path, scores: Iterable[tuple[int, float]]) -> int:
    """Persist a score batch atomically, preserving existing results and states."""
    with connect(path) as conn:
        updated = 0
        for compound_id, score in scores:
            updated += conn.execute(
                "UPDATE compound SET sa_score=?, modified_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND iteration>0 AND sa_score IS NULL",
                (score, compound_id),
            ).rowcount
        return updated


def import_synthetic_results(path, iteration, results, sa_scores):
    """Import a validated batch atomically without changing scientific states."""
    with connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        updated = 0
        for compound_id, feasible in results:
            row = conn.execute(
                "SELECT synthetic_feasibility FROM compound WHERE id=? AND iteration=?",
                (compound_id, iteration),
            ).fetchone()
            if row is None or (row[0] is not None and row[0] != feasible):
                raise RuntimeError(f"Conflicting synthetic result for compound {compound_id}.")
            updated += conn.execute(
                "UPDATE compound SET synthetic_feasibility=?, modified_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND iteration=? AND synthetic_feasibility IS NULL",
                (feasible, compound_id, iteration),
            ).rowcount
        for compound_id, score in sa_scores:
            conn.execute(
                "UPDATE compound SET sa_score=?, modified_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND iteration=? AND sa_score IS NULL",
                (score, compound_id, iteration),
            )
        return updated


def mark_gbsa_atomtyping_failures(path, compound_ids, message):
    """Preserve scores and later states when recording permanent Prime failures."""
    ids = sorted(set(compound_ids))
    if not ids:
        return 0
    marks = ",".join("?" for _ in ids)
    with connect(path) as conn:
        cursor = conn.execute(
            f"UPDATE compound SET state=?, failed_stage='gbsa', failure_message=?, "
            f"modified_at=CURRENT_TIMESTAMP WHERE id IN ({marks}) "
            "AND gbsa_score IS NULL AND state IN (?, ?)",
            [State.FAILED, message, *ids, State.DOCKED, State.GBSA_RUNNING],
        )
        return cursor.rowcount
