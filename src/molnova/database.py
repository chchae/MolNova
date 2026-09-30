"""SQLite workflow-state API."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
from molnova import _core


def connect(path: str | Path):
    return _core.open_sqlite(Path(path))


def initialize(project_toml: str | Path, schrodinger: str | Path | None = None):
    return _core.configure_project(project_toml, Path(schrodinger) if schrodinger else None)


@contextmanager
def stage_lock(path: str | Path, stage: str):
    """Prevent overlapping local invocations of one project stage."""
    lock_path = Path(f"{Path(path)}.{stage}.driver.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a")
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
    ids,
    state: str,
    failed_stage=None,
    failure_message=None,
):
    return _core.set_compound_state(
        ids,
        state,
        failed_stage,
        failure_message,
        db_path=path,
    )


def claim_compounds(
    path: str | Path,
    ids,
    expected_state: str | tuple[str, ...],
    claimed_state: str,
):
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


def find_iteration(stage: str, path: str | Path):
    return _core.find_iteration_for_state(stage, path)
