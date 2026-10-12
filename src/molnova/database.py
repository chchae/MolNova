"""SQLite workflow-state API."""
import sqlite3
import math
import time
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


_PRE_DOCKING_STATES = {
    State.GENERATED, State.SYNTHETIC_RUNNING, State.LIGPREP_RUNNING,
    State.LIGPREPPED, State.GLIDE_RUNNING,
}


def _iteration_rows(conn, iteration):
    return conn.execute(
        "SELECT id,state,docking_score,gbsa_score FROM compound WHERE iteration=?",
        (iteration,),
    ).fetchall()


def _docking_completion_reason(rows, target_count):
    if len(rows) < target_count:
        return f"generation incomplete ({len(rows)}/{target_count} compounds)"
    unfinished = sum(state in _PRE_DOCKING_STATES for _, state, _, _ in rows)
    if unfinished:
        return f"{unfinished} compounds still awaiting LigPrep/Glide completion"
    return None


def docking_completion_reason(path, iteration, target_count):
    """Wait for the entire iteration, including retryable upstream work."""
    with connect(path) as conn:
        return _docking_completion_reason(_iteration_rows(conn, iteration), target_count)


def stage_is_running(path, stage):
    try:
        with stage_lock(path, stage):
            return False
    except RuntimeError:
        return True


def upstream_completion_reason(path, iteration, target_count, stage,
                               synthetic_enabled=False):
    """Require a full iteration and finished upstream workers before starting."""
    upstream = ['generate', 'synthetic_feasibility'] if stage == 'ligprep' else ['generate', 'synthetic_feasibility', 'ligprep']
    for worker in upstream:
        if stage_is_running(path, worker):
            return f'{worker} worker still running'
    with connect(path) as conn:
        rows = _iteration_rows(conn, iteration)
        if len(rows) < target_count:
            return f'generation incomplete ({len(rows)}/{target_count} compounds)'
        blocked = {State.SYNTHETIC_RUNNING}
        if stage == 'glide':
            blocked.update((State.GENERATED, State.LIGPREP_RUNNING))
        if any(state in blocked for _, state, _, _ in rows):
            return 'upstream compounds still pending or running'
        if synthetic_enabled and conn.execute(
            'SELECT 1 FROM compound WHERE iteration=? AND state != ? '
            'AND synthetic_feasibility IS NULL LIMIT 1', (iteration, State.FAILED)
        ).fetchone():
            return 'synthetic feasibility assessment incomplete'
    return None


def reset_unsubmitted_mmgbsa(path, iteration, protected_ids=()):
    """Recover claims without a durable submission; caller holds MM-GBSA lock."""
    protected = set(protected_ids)
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        ids = [cid for (cid,) in conn.execute(
            'SELECT id FROM compound WHERE iteration=? AND state=? AND gbsa_score IS NULL',
            (iteration, State.GBSA_RUNNING)) if cid not in protected]
        for cid in ids:
            conn.execute('UPDATE compound SET state=?, modified_at=CURRENT_TIMESTAMP WHERE id=?',
                         (State.REFERENCE if iteration == 0 else State.DOCKED, cid))
        return ids


def amber_keep_count(args):
    return getattr(args, 'amber_keep_count', 500) if getattr(args, 'amber_gbsa_enabled', False) else None


def _amber_rows(conn, iteration, keep_count=None):
    rows = conn.execute(
        'SELECT id,name,state,docking_score,amber_gbsa_score,gbsa_score,fep_score,failed_stage '
        'FROM compound WHERE iteration=? AND (iteration=0 OR docking_score IS NOT NULL) ORDER BY id',
        (iteration,),
    ).fetchall()
    if iteration == 0 or keep_count is None:
        return rows
    # Rank the final Glide population before excluding scored/failed/claimed
    # rows: they retain their positions and cannot trigger lower-ranked refill.
    return sorted((r for r in rows if math.isfinite(r[3])),
                  key=lambda r: (r[3], r[0]))[:keep_count]


def _amber_completion_reason(conn, iteration, target_count, keep_count=500):
    reason = _docking_completion_reason(_iteration_rows(conn, iteration), target_count) if iteration > 0 else None
    if reason:
        return reason
    running = sum(r[2] == State.AMBER_RUNNING for r in _amber_rows(conn, iteration))
    if running:
        # A saved legacy all-pose run may include compounds outside the new
        # Glide shortlist. Its claims still require recovery, never abandonment.
        return f'{running} compounds still running Amber GBSA screening'
    rows = _amber_rows(conn, iteration, keep_count)
    pending = sum(
        state == State.AMBER_RUNNING or
        (state != State.FAILED and failed_stage != 'amber_gbsa' and (score is None or not math.isfinite(score)))
        for _, _, state, _, score, _, _, failed_stage in rows
    )
    return f'{pending} compounds await Amber GBSA screening' if pending else None


def amber_completion_reason(path, iteration, target_count, keep_count=500):
    with connect(path) as conn:
        return _amber_completion_reason(conn, iteration, target_count, keep_count)


def _prime_ranked_rows(conn, iteration, rows, amber_keep_count):
    if amber_keep_count is None or iteration == 0:
        return sorted((r for r in rows if r[2] is not None), key=lambda r: (r[2], r[0]))
    scores = {r[0]: r[4] for r in _amber_rows(conn, iteration, amber_keep_count)
              if r[4] is not None and math.isfinite(r[4])}
    return sorted((r for r in rows if r[0] in scores),
                  key=lambda r: (scores[r[0]], r[2], r[0]))[:amber_keep_count]


def prime_selection_rows(path, iteration, count, amber_keep_count=None):
    with connect(path) as conn:
        ranked = _prime_ranked_rows(conn, iteration, _iteration_rows(conn, iteration), amber_keep_count)
        names = dict(conn.execute('SELECT id,name FROM compound WHERE iteration=?', (iteration,)))
        return [(r[0], names[r[0]], r[2]) for r in ranked[:count]]


def claim_amber_candidates(path, iteration, target_count, keep_count=500):
    """Atomically rank the final Glide top-N, then claim its unscored members."""
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        if iteration > 0 and _docking_completion_reason(_iteration_rows(conn, iteration), target_count):
            return []
        rows = [r for r in _amber_rows(conn, iteration, keep_count)
                if r[2] in ((State.REFERENCE, State.DOCKED, State.GBSA_DONE, State.FEP_DONE) if iteration == 0
                            else (State.DOCKED, State.GBSA_DONE, State.FEP_DONE)) and r[4] is None
                and r[7] != 'amber_gbsa']
        for cid, *_ in rows:
            conn.execute('UPDATE compound SET state=?,modified_at=CURRENT_TIMESTAMP WHERE id=?',
                         (State.AMBER_RUNNING, cid))
        return [r[0] for r in rows]


def reset_unsubmitted_amber(path, iteration):
    """Only call with the Amber stage lock and no durable local submission."""
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(
            'UPDATE compound SET state=CASE WHEN fep_score IS NOT NULL THEN ? '
            'WHEN gbsa_score IS NOT NULL THEN ? ELSE ? END,modified_at=CURRENT_TIMESTAMP '
            'WHERE iteration=? AND state=?',
            (State.FEP_DONE, State.GBSA_DONE, State.REFERENCE if iteration == 0 else State.DOCKED, iteration, State.AMBER_RUNNING))


def amber_rankings(path, iteration, keep_count=None):
    with connect(path) as conn:
        rows = [(r[0], r[1], r[3], r[4], r[5])
                for r in _amber_rows(conn, iteration, keep_count)
                if r[4] is not None and math.isfinite(r[4])]
        return sorted(rows, key=lambda r: (r[3], r[2] if r[2] is not None else float('-inf'), r[0]))


def amber_identity_rows(path, iteration, ids):
    wanted = set(ids)
    with connect(path) as conn:
        return [tuple(r) for r in conn.execute(
            'SELECT id,name,smiles FROM compound WHERE iteration=? ORDER BY id', (iteration,)) if r[0] in wanted]


def store_amber_results(path, iteration, results, allowed_ids):
    """Validate the whole batch before committing; never overwrite Prime/FEP scores."""
    allowed = set(allowed_ids)
    if len({r['id'] for r in results}) != len(results):
        raise ValueError('Duplicate Amber result IDs')
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        updated = 0
        for result in results:
            cid = result['id']
            if isinstance(cid, bool) or not isinstance(cid, int) or cid < 1:
                raise ValueError(f'Invalid Amber compound ID: {cid}')
            row = conn.execute(
                'SELECT amber_gbsa_score,state,gbsa_score,fep_score FROM compound WHERE id=? AND iteration=?',
                (cid, iteration),
            ).fetchone()
            if cid not in allowed or row is None:
                raise ValueError(f'Unsubmitted Amber compound: {cid}')
            score = result.get('score')
            error = result.get('error')
            if ((score is None) == (error is None)
                    or (score is not None and (isinstance(score, bool)
                        or not isinstance(score, (float, int)) or not math.isfinite(score)))
                    or (error is not None and (not isinstance(error, str) or not error.strip()))):
                raise ValueError(f'Invalid Amber result: {cid}')
            if row[0] is not None and score is not None and row[0] != score:
                raise ValueError(f'Conflicting Amber score: {cid}')
            if row[0] is not None:
                continue
            state = State.FEP_DONE if row[3] is not None else State.GBSA_DONE if row[2] is not None else State.REFERENCE if iteration == 0 else State.DOCKED
            if score is not None:
                updated += conn.execute(
                    'UPDATE compound SET amber_gbsa_score=?,state=CASE WHEN state=? THEN ? ELSE state END,'
                    'modified_at=CURRENT_TIMESTAMP WHERE id=?',
                    (score, State.AMBER_RUNNING, state, cid),
                ).rowcount
            else:
                # Terminal parameterization/calculation failures cannot supply a ranking score.
                conn.execute('UPDATE compound SET state=?,failed_stage=?,failure_message=?,'
                             'modified_at=CURRENT_TIMESTAMP WHERE id=? AND state=?',
                             (state if row[2] is not None else State.FAILED, 'amber_gbsa', error,
                              cid, State.AMBER_RUNNING))
        return updated


def _mmgbsa_candidates(conn, iteration, count, target_count, amber_keep_count=None):
    rows = _iteration_rows(conn, iteration)
    if iteration == 0:
        if amber_keep_count is not None and _amber_completion_reason(conn, iteration, target_count, amber_keep_count):
            return []
        return [cid for cid, state, _, score in rows
                if score is None and state in (State.REFERENCE, State.DOCKED)]
    if iteration < 0 or _docking_completion_reason(rows, target_count) is not None:
        return []
    if amber_keep_count is not None and _amber_completion_reason(conn, iteration, target_count, amber_keep_count):
        return []
    docked = _prime_ranked_rows(conn, iteration, rows, amber_keep_count)
    return [cid for cid, state, _, score in docked[:count]
            if score is None and state == State.DOCKED]


def mmgbsa_candidates(path: str | Path, iteration: int, count: int,
                      target_count: int = 1, amber_keep_count=None) -> list[int]:
    """Select unscored final top-N only after all upstream work has finished."""
    with connect(path) as conn:
        return _mmgbsa_candidates(conn, iteration, count, target_count, amber_keep_count)


def claim_mmgbsa_candidates(path, iteration, count, target_count, amber_keep_count=None):
    """Check completion, rank and claim in one SQLite write transaction."""
    with connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        ids = _mmgbsa_candidates(conn, iteration, count, target_count, amber_keep_count)
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            marks = ",".join("?" for _ in batch)
            conn.execute(
                f"UPDATE compound SET state=?, failed_stage=NULL, failure_message=NULL, "
                f"modified_at=CURRENT_TIMESTAMP WHERE id IN ({marks})",
                [State.GBSA_RUNNING, *batch],
            )
        return ids


def iteration_completion_reason(path, iteration, target_count, input_count, elite_count, amber_keep_count=None):
    """Return a blocking reason until docking and all selected GBSA work finish.

    Terminal failures count as processed, but cannot supply an elite. Docked
    compounds outside the final top-N need no GBSA result. A single SQLite read
    keeps the readiness decision consistent with the persisted workflow state.
    """
    with connect(path) as conn:
        rows = _iteration_rows(conn, iteration)
        if amber_keep_count is not None:
            reason = _amber_completion_reason(conn, iteration, target_count, amber_keep_count)
            if reason:
                return reason
        docked = _prime_ranked_rows(conn, iteration, rows, amber_keep_count)
    reason = _docking_completion_reason(rows, target_count)
    if reason is not None:
        return reason
    running = sum(state == State.GBSA_RUNNING for _, state, _, _ in rows)
    if running:
        return f"MM-GBSA still running for {running} compounds"
    pending = sum(
        state != State.FAILED and (score is None or not math.isfinite(score))
        for _, state, _, score in docked[:input_count]
    )
    if pending:
        ranking = 'Amber' if amber_keep_count is not None else 'docking'
        return f"final {ranking} top-{input_count} has {pending} unfinished MM-GBSA results"
    valid = sum(score is not None and math.isfinite(score) for _, _, _, score in rows)
    if valid < elite_count:
        return f"only {valid}/{elite_count} valid MM-GBSA results for elite selection"
    return None


def iteration_gbsa_elites(path, iteration, count):
    """Select lowest finite GBSA scores from exactly the completed iteration."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT id,name,smiles,iteration,docking_score,gbsa_score FROM compound "
            "WHERE iteration=? AND gbsa_score IS NOT NULL "
            "ORDER BY gbsa_score ASC, docking_score ASC, id ASC",
            (iteration,),
        ).fetchall()
    return [row for row in rows if math.isfinite(row[5])][:count]


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


def mmgbsa_submission_rows(path, compound_ids):
    """Inspect saved submission identity against the current project database."""
    ids = sorted(set(compound_ids))
    rows = {}
    with connect(path) as conn:
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            marks = ",".join("?" for _ in batch)
            for cid, iteration, state, score in conn.execute(
                f"SELECT id,iteration,state,gbsa_score FROM compound WHERE id IN ({marks})", batch
            ):
                rows[cid] = (iteration, state, score)
    return rows


def has_running_mmgbsa(path, iteration):
    with connect(path) as conn:
        return conn.execute('SELECT 1 FROM compound WHERE iteration=? AND state=? LIMIT 1',
                            (iteration, State.GBSA_RUNNING)).fetchone() is not None


def stage_completion_reason(path, iteration, stage, target_count, input_count,
                            synthetic_enabled=False, amber_keep_count=None):
    """Driver readiness from persisted state, never worker exit status alone."""
    if stage == 'mmgbsa-recovery':
        return 'Existing Prime claims still running' if has_running_mmgbsa(path, iteration) else None
    if stage == 'amber-gbsa':
        return amber_completion_reason(path, iteration, target_count,
                                       amber_keep_count if amber_keep_count is not None else 500)
    if stage == 'mmgbsa' and iteration == 0:
        return reference_mmgbsa_completion_reason(path)
    if stage == 'mmgbsa':
        # Ending a single iteration does not require enough elites for N+1.
        return iteration_completion_reason(path, iteration, target_count, input_count, 0, amber_keep_count)
    if stage == 'fep':
        # Current FEP stage exports candidates; no automated calculation exists.
        return None
    with connect(path) as conn:
        rows = _iteration_rows(conn, iteration)
        if len(rows) < target_count:
            return f'generation incomplete ({len(rows)}/{target_count} compounds)'
        if stage == 'generate':
            return None
        if stage == 'synthetic':
            pending = conn.execute(
                'SELECT COUNT(*) FROM compound WHERE iteration=? AND state != ? '
                'AND (synthetic_feasibility IS NULL OR sa_score IS NULL OR state=?)',
                (iteration, State.FAILED, State.SYNTHETIC_RUNNING),
            ).fetchone()[0]
            return f'{pending} compounds await synthetic assessment' if pending else None
        if stage == 'ligprep':
            blocked = {State.GENERATED, State.SYNTHETIC_RUNNING, State.LIGPREP_RUNNING}
            pending = sum(state in blocked for _, state, _, _ in rows)
            if pending:
                return f'{pending} compounds await LigPrep completion'
            if synthetic_enabled and conn.execute(
                'SELECT 1 FROM compound WHERE iteration=? AND state != ? '
                'AND synthetic_feasibility IS NULL LIMIT 1', (iteration, State.FAILED)
            ).fetchone():
                return 'synthetic assessment incomplete'
            return None
        if stage == 'glide':
            return _docking_completion_reason(rows, target_count)
    raise ValueError(f'Unknown stage: {stage}')



def start_iteration_timer(path, iteration):
    """Persist the original start across driver restarts; preserve old results."""
    now = time.time()
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(
            "CREATE TABLE IF NOT EXISTS iteration_timing ("
            "iteration INTEGER PRIMARY KEY, started_at REAL NOT NULL, "
            "finished_at REAL, start_source TEXT NOT NULL)"
        )
        existing = conn.execute(
            'SELECT started_at FROM iteration_timing WHERE iteration=?', (iteration,)
        ).fetchone()
        if existing:
            return existing[0]
        # Existing iterations predate the timer: report an explicitly estimated
        # duration from their earliest compound creation (SQLite UTC timestamps).
        earliest = None
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='compound'").fetchone():
            earliest = conn.execute(
                "SELECT MIN(CAST(strftime('%s', created_at) AS REAL)) "
                "FROM compound WHERE iteration=?", (iteration,)
            ).fetchone()[0]
        started = min(now, earliest) if earliest is not None else now
        conn.execute(
            'INSERT INTO iteration_timing(iteration,started_at,start_source) VALUES (?,?,?)',
            (iteration, started, 'compound_created_at' if earliest is not None else 'driver'),
        )
        return started


def finish_iteration_timer(path, iteration):
    """Checkpoint verified completion once; include wait/recovery/downtime."""
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(
            'UPDATE iteration_timing SET finished_at=COALESCE(finished_at, ?) WHERE iteration=?',
            (time.time(), iteration),
        )
        row = conn.execute(
            'SELECT started_at,finished_at,start_source FROM iteration_timing WHERE iteration=?',
            (iteration,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f'Iteration {iteration} has no start time')
        return max(0, row[1] - row[0]), row[2] == 'compound_created_at'



def reference_mmgbsa_completion_reason(path):
    """All references need a score or terminal failure, independent of top-N."""
    with connect(path) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='compound'").fetchone():
            return None
        rows = _iteration_rows(conn, 0)
    pending = sum(state != State.FAILED and score is None for _, state, _, score in rows)
    running = sum(state == State.GBSA_RUNNING for _, state, _, _ in rows)
    if running:
        return f'reference MM-GBSA still running for {running} compounds'
    return f'{pending} reference compounds await MM-GBSA results' if pending else None
