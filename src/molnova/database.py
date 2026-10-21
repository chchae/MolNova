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


RDK_DESCRIPTOR_TYPES = {
    'rdk_sa_score': 'REAL', 'rdk_qed': 'REAL', 'rdk_mw': 'REAL',
    'rdk_clogp': 'REAL', 'rdk_tpsa': 'REAL', 'rdk_hbd': 'INTEGER',
    'rdk_hba': 'INTEGER', 'rdk_rotatable_bonds': 'INTEGER',
    'rdk_formal_charge': 'INTEGER', 'rdk_heavy_atoms': 'INTEGER',
    'rdk_aromatic_rings': 'INTEGER', 'rdk_fraction_csp3': 'REAL',
    'rdk_ro5_violations': 'INTEGER', 'rdk_veber_pass': 'INTEGER',
    'rdk_pains_alerts': 'INTEGER', 'rdk_brenk_alerts': 'INTEGER',
}


def ensure_rdkit_descriptor_columns(conn):
    """Add nullable properties without replacing rows, results or active claims."""
    columns = {row[1] for row in conn.execute('PRAGMA table_info(compound)')}
    for name, sql_type in RDK_DESCRIPTOR_TYPES.items():
        if name not in columns:
            conn.execute(f'ALTER TABLE compound ADD COLUMN {name} {sql_type}')
    if 'sa_score' in columns:
        conn.execute('UPDATE compound SET rdk_sa_score=sa_score '
                     'WHERE rdk_sa_score IS NULL AND sa_score BETWEEN 1 AND 10')
        conn.execute('UPDATE compound SET sa_score=rdk_sa_score '
                     'WHERE sa_score IS NULL AND rdk_sa_score BETWEEN 1 AND 10')


def connect(path: str | Path) -> sqlite3.Connection:
    return _core.open_sqlite(Path(path))


def distribution_snapshot(path, iteration=None):
    """Read one consistent diagnostic snapshot, without migrations or claims."""
    import json

    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA query_only=ON')
        conn.execute('PRAGMA busy_timeout=30000')
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('BEGIN')
        columns = {row[1] for row in conn.execute('PRAGMA table_info(compound)')}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        wanted = ['id', 'name', 'smiles', 'iteration', 'state', 'failed_stage',
                  'docking_score', 'amber_gbsa_score', 'gbsa_score', 'fep_score',
                  'sa_score', *RDK_DESCRIPTOR_TYPES]
        fields = ','.join(f'c.{name}' if name in columns else f'NULL AS {name}' for name in wanted)
        selected = 's.passed' if 'compound_druglike_selection' in tables else 'NULL'
        join = ('LEFT JOIN compound_druglike_selection s ON s.compound_id=c.id'
                if 'compound_druglike_selection' in tables else '')
        rows = [dict(row) for row in conn.execute(
            f'SELECT {fields}, {selected} AS druglike_selected FROM compound c {join} '
            'WHERE c.iteration>=0 AND (? IS NULL OR c.iteration<=?) ORDER BY c.iteration,c.id',
            (iteration, iteration))]
        scaffolds = {}
        if 'iteration_scaffold' in tables:
            scaffolds = {row[0]: json.loads(row[1]) for row in conn.execute(
                'SELECT iteration,snapshot FROM iteration_scaffold '
                'WHERE ? IS NULL OR iteration<=? ORDER BY iteration', (iteration, iteration))}
        return rows, scaffolds
    finally:
        conn.close()


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
    upstream = ['generate', 'druglike', 'synthetic_feasibility'] if stage == 'ligprep' else ['generate', 'druglike', 'synthetic_feasibility', 'ligprep']
    for worker in upstream:
        if stage_is_running(path, worker):
            return f'{worker} worker still running'
    reason = druglike_completion_reason(path, iteration)
    if reason:
        return reason
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
            'SELECT 1 FROM compound WHERE iteration=? AND state NOT IN (?,?,?) '
            'AND synthetic_feasibility IS NULL LIMIT 1', (iteration, State.FAILED, State.SA_REJECTED, State.DRUGLIKE_REJECTED)
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


def _remember_amber_failure(conn, cid, message):
    conn.execute('CREATE TABLE IF NOT EXISTS amber_failure ('
                 'compound_id INTEGER PRIMARY KEY REFERENCES compound(id),message TEXT NOT NULL)')
    conn.execute('INSERT INTO amber_failure VALUES (?,?) ON CONFLICT(compound_id) DO NOTHING',
                 (cid, message or 'Amber calculation failed'))


def _amber_failure_ids(conn, iteration):
    ids = {row[0] for row in conn.execute(
        "SELECT id FROM compound WHERE iteration=? AND failed_stage='amber_gbsa'", (iteration,))}
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='amber_failure'").fetchone():
        ids.update(row[0] for row in conn.execute('SELECT c.id FROM compound c JOIN amber_failure f '
                                                 'ON f.compound_id=c.id WHERE c.iteration=?', (iteration,)))
    return ids


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
    failed_ids = _amber_failure_ids(conn, iteration)
    pending = sum(
        state == State.AMBER_RUNNING or
        (state != State.FAILED and cid not in failed_ids and (score is None or not math.isfinite(score)))
        for cid, _, state, _, score, _, _, failed_stage in rows
    )
    return f'{pending} compounds await Amber GBSA screening' if pending else None


def amber_completion_reason(path, iteration, target_count, keep_count=500):
    with connect(path) as conn:
        return _amber_completion_reason(conn, iteration, target_count, keep_count)


def _prime_ranked_rows(conn, iteration, rows, amber_keep_count):
    docking = sorted((r for r in rows if r[2] is not None and math.isfinite(r[2])), key=lambda r: (r[2], r[0]))
    if iteration == 0:
        return docking
    scores = {r[0]: r[4] for r in _amber_rows(conn, iteration, amber_keep_count)
              if r[4] is not None and math.isfinite(r[4])}
    if not scores:
        return docking
    return sorted((r for r in rows if r[0] in scores),
                  key=lambda r: (scores[r[0]], r[2], r[0]))


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
        failed_ids = _amber_failure_ids(conn, iteration)
        rows = [r for r in _amber_rows(conn, iteration, keep_count)
                if r[2] in ((State.REFERENCE, State.DOCKED, State.GBSA_DONE, State.FEP_DONE) if iteration == 0
                            else (State.DOCKED, State.GBSA_DONE, State.FEP_DONE)) and r[4] is None
                and r[0] not in failed_ids]
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
                _remember_amber_failure(conn, cid, error)
                conn.execute('UPDATE compound SET state=?,failed_stage=?,failure_message=?,'
                             'modified_at=CURRENT_TIMESTAMP WHERE id=? AND state=?',
                             (state if row[2] is not None else State.FAILED, 'amber_gbsa', error,
                              cid, State.AMBER_RUNNING))
        return updated


def _mmgbsa_candidates(conn, iteration, count, target_count, amber_keep_count=None):
    rows = _iteration_rows(conn, iteration)
    if any(state == State.AMBER_RUNNING for _, state, _, _ in rows):
        return []
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
    amber_failed = {row[0] for row in conn.execute(
        "SELECT id FROM compound WHERE iteration=? AND state=? AND failed_stage='amber_gbsa'",
        (iteration, State.FAILED))}
    return [cid for cid, state, _, score in docked[:count]
            if score is None and (state == State.DOCKED or cid in amber_failed)]


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
            for cid, message in conn.execute(
                    f"SELECT id,failure_message FROM compound WHERE id IN ({marks}) AND failed_stage='amber_gbsa'", batch):
                _remember_amber_failure(conn, cid, message)
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
        running_amber = sum(state == State.AMBER_RUNNING for _, state, _, _ in rows)
        if running_amber:
            return f'{running_amber} compounds still running Amber GBSA screening'
        if amber_keep_count is not None:
            reason = _amber_completion_reason(conn, iteration, target_count, amber_keep_count)
            if reason:
                return reason
        docked = _prime_ranked_rows(conn, iteration, rows, amber_keep_count)
        amber_failed = {row[0] for row in conn.execute(
            "SELECT id FROM compound WHERE iteration=? AND state=? AND failed_stage='amber_gbsa'",
            (iteration, State.FAILED))}
    reason = _docking_completion_reason(rows, target_count)
    if reason is not None:
        return reason
    running = sum(state == State.GBSA_RUNNING for _, state, _, _ in rows)
    if running:
        return f"MM-GBSA still running for {running} compounds"
    pending = sum(
        (state != State.FAILED or cid in amber_failed) and (score is None or not math.isfinite(score))
        for cid, state, _, score in docked[:input_count]
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


def iteration_scaffold(path, iteration, snapshot=None):
    """Load or atomically initialize an immutable automatic iteration core."""
    import json

    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(
            'CREATE TABLE IF NOT EXISTS iteration_scaffold ('
            'iteration INTEGER PRIMARY KEY, snapshot TEXT NOT NULL)'
        )
        if snapshot is not None:
            conn.execute(
                'INSERT INTO iteration_scaffold(iteration,snapshot) VALUES (?,?) '
                'ON CONFLICT(iteration) DO NOTHING',
                (iteration, json.dumps(snapshot, sort_keys=True)),
            )
        row = conn.execute(
            'SELECT snapshot FROM iteration_scaffold WHERE iteration=?', (iteration,),
        ).fetchone()
    return json.loads(row[0]) if row else None


def iteration_has_compounds(path, iteration):
    with connect(path) as conn:
        return conn.execute(
            'SELECT 1 FROM compound WHERE iteration=? LIMIT 1', (iteration,),
        ).fetchone() is not None


def iteration_has_saved_work(path, iteration):
    """Durable work/preparation pins iteration files; timing alone does not."""
    with connect(path) as conn:
        for table in ('compound', 'iteration_scaffold', 'iteration_combichem',
                      'iteration_combichem_history', 'iteration_sa_selection', 'combichem_batch',
                      'iteration_workspace', 'iteration_druglike_selection', 'iteration_fep_selection'):
            if (conn.execute('SELECT 1 FROM sqlite_master WHERE type=? AND name=?',
                             ('table', table)).fetchone()
                    and conn.execute(f'SELECT 1 FROM {table} WHERE iteration=? LIMIT 1',
                                     (iteration,)).fetchone()):
                return True
    return False


def _deleted_iteration(conn, iteration):
    if conn.execute('SELECT 1 FROM compound WHERE iteration=? LIMIT 1', (iteration,)).fetchone():
        return False
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ('iteration_druglike_selection', 'iteration_sa_selection', 'combichem_compound_origin'):
        if table in tables and conn.execute(f'SELECT 1 FROM {table} WHERE iteration=? LIMIT 1',
                                            (iteration,)).fetchone():
            return True
    return ('combichem_batch' in tables and conn.execute(
        'SELECT 1 FROM combichem_batch WHERE iteration=? AND inserted>0 LIMIT 1',
        (iteration,)).fetchone() is not None)


def iteration_was_deleted(path, iteration):
    """Zero rows plus evidence of prior compounds; a new zero-row plan is a resume."""
    with connect(path) as conn:
        return _deleted_iteration(conn, iteration)


def reset_deleted_iteration(path, iteration, backup, report_ids=()):
    """Under all stage locks, back up SQLite/WAL and reset only a deleted iteration.

    The caller verifies native jobs and archives working files. Membership IDs
    must come from saved combichem origins or the original druglike report, never
    from guessed SQLite ID ranges. Existing compound rows are never removed.
    """
    if iteration < 1:
        raise ValueError('Cannot reset iteration zero')
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not _deleted_iteration(conn, iteration):
            raise RuntimeError('Iteration is no longer empty/deleted; files and database retained')
        if conn.execute('SELECT 1 FROM compound WHERE iteration>? LIMIT 1', (iteration,)).fetchone():
            raise RuntimeError('Later iteration compounds depend on this iteration; reset blocked')
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ids = {row[0] for row in conn.execute(
            'SELECT compound_id FROM combichem_compound_origin WHERE iteration=?', (iteration,))
               } if 'combichem_compound_origin' in tables else set()
        reported = set(report_ids)
        if reported:
            if ids and reported != ids:
                raise RuntimeError('Druglike report and saved compound origins disagree; reset blocked')
            ids = reported
        saved = conn.execute('SELECT population_count FROM iteration_druglike_selection WHERE iteration=?',
                             (iteration,)).fetchone() if 'iteration_druglike_selection' in tables else None
        if saved and len(ids) != saved[0]:
            raise RuntimeError('Cannot identify deleted selection members; original druglike report '
                               'or compound origins required before reset')
        conn.execute('CREATE TEMP TABLE reset_compound_ids(id INTEGER PRIMARY KEY)')
        conn.executemany('INSERT INTO reset_compound_ids VALUES (?)', ((cid,) for cid in ids))
        if conn.execute('SELECT 1 FROM compound WHERE id IN (SELECT id FROM reset_compound_ids) '
                        'OR similar_to IN (SELECT id FROM reset_compound_ids) LIMIT 1').fetchone():
            raise RuntimeError('Deleted iteration IDs are referenced by current compounds; reset blocked')
        backup = Path(backup)
        if backup.exists():
            raise FileExistsError(f'Refusing to overwrite iteration backup: {backup}')
        # Use a separate reader: backing up the connection holding a write
        # transaction would wait on itself. SQLite backup includes committed WAL.
        with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(backup) as target:
                source.backup(target)
        for table in ('compound_druglike_selection', 'amber_failure'):
            if table in tables:
                conn.execute(f'DELETE FROM {table} WHERE compound_id IN (SELECT id FROM reset_compound_ids)')
        for table in ('combichem_compound_origin', 'combichem_batch', 'iteration_combichem_history',
                      'iteration_combichem', 'iteration_druglike_selection', 'iteration_sa_selection',
                      'iteration_fep_selection', 'iteration_scaffold', 'iteration_workspace'):
            if table in tables:
                conn.execute(f'DELETE FROM {table} WHERE iteration=?', (iteration,))
        # The supervising driver already owns timing rows; restart their clocks
        # rather than deleting rows it will finish after the worker exits.
        now = time.time()
        if 'iteration_timing' in tables:
            conn.execute("UPDATE iteration_timing SET started_at=?,finished_at=NULL,start_source='driver' "
                         'WHERE iteration=?', (now, iteration))
        if 'stage_timing' in tables:
            conn.execute('DELETE FROM stage_timing WHERE iteration=? AND stage!=?', (iteration, 'generate'))
            conn.execute('UPDATE stage_timing SET started_at=?,finished_at=NULL WHERE iteration=?',
                         (now, iteration))
        _mark_iteration_workspace_prepared(conn, iteration)


def mark_iteration_workspace_prepared(path, iteration):
    """Checkpoint cleanup before generation so zero-row retries keep their files."""
    with connect(path) as conn:
        _mark_iteration_workspace_prepared(conn, iteration)


def _mark_iteration_workspace_prepared(conn, iteration):
    conn.execute('CREATE TABLE IF NOT EXISTS iteration_workspace ('
                 'iteration INTEGER PRIMARY KEY, prepared_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)')
    columns = {row[1] for row in conn.execute('PRAGMA table_info(iteration_workspace)')}
    if 'generation_policy' not in columns:
        conn.execute("ALTER TABLE iteration_workspace ADD COLUMN generation_policy TEXT NOT NULL DEFAULT 'legacy'")
    conn.execute("INSERT INTO iteration_workspace(iteration,generation_policy) VALUES (?,'raw_v1') ON CONFLICT(iteration) DO NOTHING",
                 (iteration,))

def _uses_raw_generation(conn, iteration):
    columns = {row[1] for row in conn.execute('PRAGMA table_info(iteration_workspace)')}
    if 'generation_policy' not in columns:
        return False
    row = conn.execute('SELECT generation_policy FROM iteration_workspace WHERE iteration=?', (iteration,)).fetchone()
    return row is not None and row[0] == 'raw_v1'


def uses_raw_generation(path, iteration):
    with connect(path) as conn:
        return _uses_raw_generation(conn, iteration)


def _validate_druglike_population(conn, iteration):
    """Never reinterpret deleted/replaced rows as an unfinished generation."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_druglike_selection'").fetchone():
        return None
    saved = conn.execute('SELECT population_count FROM iteration_druglike_selection WHERE iteration=?',
                         (iteration,)).fetchone()
    if saved is None:
        return None
    actual = conn.execute('SELECT COUNT(*) FROM compound WHERE iteration=?', (iteration,)).fetchone()[0]
    selected = conn.execute('SELECT COUNT(*) FROM compound c JOIN compound_druglike_selection s '
                            'ON s.compound_id=c.id WHERE c.iteration=?', (iteration,)).fetchone()[0]
    if actual != saved[0] or selected != actual:
        raise RuntimeError(
            f'Iteration {iteration}: frozen drug likeness population is inconsistent '
            f'(expected={saved[0]}, compounds={actual}, selection members={selected}). '
            'Compound rows may have been manually deleted or replaced. '
            'Restore the original database or explicitly reset the whole iteration, '
            'including selection records, combichem progress and calculation files; '
            'automatic regeneration is blocked to preserve results and saved jobs.')
    return saved[0]


def validate_generation_population(path, iteration, target_count=None):
    """Fail before expensive sampling when a frozen population cannot be resumed."""
    with connect(path) as conn:
        count = _validate_druglike_population(conn, iteration)
    if count is not None and target_count is not None and target_count > count:
        raise RuntimeError(f'Iteration {iteration}: cannot increase target-count to {target_count} '
                           f'after drug likeness froze {count} compounds; restore the original '
                           'target-count or start a new iteration.')


def druglike_completion_reason(path, iteration):
    with connect(path) as conn:
        _validate_druglike_population(conn, iteration)
        if not _uses_raw_generation(conn, iteration):
            return None
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_druglike_selection'").fetchone():
            if conn.execute('SELECT 1 FROM iteration_druglike_selection WHERE iteration=?', (iteration,)).fetchone():
                return None
        return 'iteration-wide drug likeness selection incomplete'


def druglike_population(path, iteration):
    with connect(path) as conn:
        return [dict(id=cid, smiles=smiles, qed=qed, sa_score=sa, ro5_violations=ro5)
                for cid, smiles, qed, sa, ro5 in conn.execute(
                    'SELECT id,smiles,rdk_qed,rdk_sa_score,rdk_ro5_violations '
                    'FROM compound WHERE iteration=? ORDER BY id', (iteration,))]


def druglike_report(path, iteration):
    with connect(path) as conn:
        return conn.execute('SELECT c.id,c.smiles,c.rdk_qed,c.rdk_sa_score,c.rdk_ro5_violations,s.passed,s.reason '
                            'FROM compound c JOIN compound_druglike_selection s ON s.compound_id=c.id '
                            'WHERE c.iteration=? ORDER BY c.id', (iteration,)).fetchall()


def apply_druglike_selection(path, iteration, target_count, selections, sa_ratio=1.0):
    """Freeze the full population and every exclusion in the same transaction."""
    from molnova.config import sa_accepted_ratio
    sa_ratio = sa_accepted_ratio({'sa-accepted-ratio': sa_ratio})
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('CREATE TABLE IF NOT EXISTS iteration_druglike_selection ('
                     'iteration INTEGER PRIMARY KEY,population_count INTEGER NOT NULL,'
                     'accepted_count INTEGER NOT NULL,sa_ratio REAL NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS compound_druglike_selection ('
                     'compound_id INTEGER PRIMARY KEY REFERENCES compound(id),'
                     'passed INTEGER NOT NULL CHECK(passed IN (0,1)),reason TEXT)')
        saved = conn.execute('SELECT population_count,accepted_count,sa_ratio '
                             'FROM iteration_druglike_selection WHERE iteration=?', (iteration,)).fetchone()
        if saved:
            _validate_druglike_population(conn, iteration)
            return saved
        if not _uses_raw_generation(conn, iteration):
            return None
        rows = conn.execute('SELECT id,state FROM compound WHERE iteration=? ORDER BY id', (iteration,)).fetchall()
        if len(rows) < target_count:
            raise RuntimeError('Drug likeness waits for the complete generated population')
        if any(state != State.GENERATED for _, state in rows):
            raise RuntimeError('Drug likeness cannot change an iteration with downstream work')
        if [cid for cid, _ in rows] != [row['id'] for row in selections]:
            raise RuntimeError('Drug likeness population changed before commit')
        chosen = [row for row in selections if row['druglike_selected']]
        chosen.sort(key=lambda row: (row['sa_score'], row['id']))
        sa_ids = {row['id'] for row in chosen[:math.ceil(len(chosen) * sa_ratio)]}
        for row in selections:
            passed = row['druglike_selected'] and row['id'] in sa_ids
            reason = row['druglike_reject_reason'] if not row['druglike_selected'] else (None if passed else 'sa_rank')
            conn.execute('INSERT INTO compound_druglike_selection VALUES (?,?,?)', (row['id'], int(passed), reason))
            if not passed:
                conn.execute('UPDATE compound SET state=?,modified_at=CURRENT_TIMESTAMP WHERE id=?',
                             (State.DRUGLIKE_REJECTED, row['id']))
        conn.execute('INSERT INTO iteration_druglike_selection VALUES (?,?,?,?)',
                     (iteration, len(rows), len(sa_ids), sa_ratio))
        return len(rows), len(sa_ids), sa_ratio


def combichem_plan(path, iteration, plan=None):
    """Persist immutable pools/parameters separately from the atomic cursor."""
    import json
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        # TEXT cursors also support products larger than SQLite's 64-bit range.
        conn.execute('CREATE TABLE IF NOT EXISTS iteration_combichem ('
                     'iteration INTEGER PRIMARY KEY, plan TEXT NOT NULL, next_index TEXT NOT NULL)')
        if plan is not None:
            conn.execute('INSERT INTO iteration_combichem(iteration,plan,next_index) VALUES (?,?,?) '
                         'ON CONFLICT(iteration) DO NOTHING',
                         (iteration, json.dumps(plan, sort_keys=True), '0'))
        row = conn.execute('SELECT plan,next_index FROM iteration_combichem WHERE iteration=?',
                           (iteration,)).fetchone()
    return dict(json.loads(row[0]), next_index=int(row[1])) if row else None


def _combichem_empty_batches(conn, iteration, revision):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='combichem_batch'").fetchone():
        return 0
    count = 0
    for (inserted,) in conn.execute(
        'SELECT inserted FROM combichem_batch WHERE iteration=? AND revision=? ORDER BY id DESC LIMIT 10',
        (iteration, revision),
    ):
        if inserted:
            break
        count += 1
    return count


def combichem_empty_batches(path, iteration, revision=0):
    with connect(path) as conn:
        return _combichem_empty_batches(conn, iteration, revision)


def combichem_plans(path, iteration):
    """Read all tried round plans without changing their scientific provenance."""
    import json
    with connect(path) as conn:
        plans = []
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_combichem_history'").fetchone():
            plans.extend(json.loads(row[0]) for row in conn.execute(
                'SELECT plan FROM iteration_combichem_history WHERE iteration=? ORDER BY round', (iteration,)))
        plans.extend(json.loads(row[0]) for row in conn.execute(
            'SELECT plan FROM iteration_combichem WHERE iteration=?', (iteration,)))
    return plans


def combichem_compound_scaffold(path, compound_id):
    """Recover an elite's actual generation scaffold, including changed attachments."""
    import json
    with connect(path) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='combichem_compound_origin'").fetchone():
            return None
        origin = conn.execute('SELECT iteration,round FROM combichem_compound_origin WHERE compound_id=?',
                              (compound_id,)).fetchone()
        if origin is None:
            return None  # Legacy compounds used the immutable iteration snapshot.
        row = conn.execute('SELECT plan FROM iteration_combichem WHERE iteration=?', (origin[0],)).fetchone()
        plan = json.loads(row[0]) if row else None
        if plan is None or plan.get('round', 0) != origin[1]:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_combichem_history'").fetchone():
                row = conn.execute('SELECT plan FROM iteration_combichem_history WHERE iteration=? AND round=?',
                                   origin).fetchone()
                plan = json.loads(row[0]) if row else None
            else:
                plan = None
        if plan is None:
            raise RuntimeError('Combichem compound origin has no saved round; restore its provenance')
    return plan.get('attachment_selection')


def advance_combichem_plan(path, iteration, previous, plan, target_count, *, change_attachments=False):
    """Archive a completed/stalled round and atomically install the next plan."""
    import json
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT plan,next_index FROM iteration_combichem WHERE iteration=?',
                           (iteration,)).fetchone()
        if row is None:
            raise RuntimeError('Combichem refill has no saved plan')
        current = dict(json.loads(row[0]), next_index=int(row[1]))
        if current != previous:
            return current  # Another claimant already advanced this round.
        if not change_attachments and current['next_index'] != int(current['total_combinations']):
            raise RuntimeError('Cannot refill an unfinished combichem round')
        count = conn.execute('SELECT COUNT(*) FROM compound WHERE iteration=?', (iteration,)).fetchone()[0]
        if count >= target_count:
            return current
        pinned = ('library_path', 'library_sha256', 'attachment_labels',
                  'selection_weight', 'batch_size')
        if (any(plan[key] != current[key] for key in pinned)
                or plan.get('sidechain_max_heavy_atoms') != current.get('sidechain_max_heavy_atoms')):
            raise RuntimeError('Combichem refill differs from saved scaffold/library/parameters')
        revision = current.get('attachment_revision', 0)
        if change_attachments:
            if _combichem_empty_batches(conn, iteration, revision) < 10:
                raise RuntimeError('Attachment change requires 10 consecutive empty batches')
            original = conn.execute('SELECT snapshot FROM iteration_scaffold WHERE iteration=?', (iteration,)).fetchone()
            original = json.loads(original[0]) if original else {}
            selection = plan.get('attachment_selection', {})
            if (original.get('source') != 'single_elite_sidechain_replacement'
                    or plan.get('attachment_revision') != revision + 1
                    or plan.get('base_scaffold_smiles') != original.get('scaffold_smiles')
                    or selection.get('source') != original.get('source')
                    or selection.get('source_ids') != original.get('source_ids')
                    or selection.get('replacement_elite_smiles') != original.get('replacement_elite_smiles')
                    or selection.get('replacement_from_iteration') != original.get('replacement_from_iteration')
                    or selection.get('replacement_candidates') != original.get('replacement_candidates')
                    or selection.get('scaffold_smiles') != plan['scaffold_smiles']):
                raise RuntimeError('Invalid combichem attachment-change provenance')
            groups = original['replacement_candidates']
            if (len(selection['replacement_cuts']) != len(groups)
                    or any(cut not in group for cut, group in zip(selection['replacement_cuts'], groups))):
                raise RuntimeError('Invalid combichem attachment cuts')
            signature = [cut['bond_index'] for cut in selection['replacement_cuts']]
            tried = [original]
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_combichem_history'").fetchone():
                tried.extend(json.loads(row[0]).get('attachment_selection', original) for row in conn.execute(
                    'SELECT plan FROM iteration_combichem_history WHERE iteration=?', (iteration,)))
            tried.append(current.get('attachment_selection', original))
            if any([cut['bond_index'] for cut in prior['replacement_cuts']] == signature for prior in tried):
                raise RuntimeError('Combichem attachment combination was already tried')
        elif (plan['scaffold_smiles'] != current['scaffold_smiles']
              or plan.get('attachment_revision', 0) != revision
              or plan.get('attachment_selection') != current.get('attachment_selection')):
            raise RuntimeError('Combichem refill differs from saved scaffold/library/parameters')
        round_id = current.get('round', 0)
        following = dict(plan, round=round_id + 1)
        following.pop('next_index', None)
        conn.execute('CREATE TABLE IF NOT EXISTS iteration_combichem_history ('
                     'iteration INTEGER NOT NULL, round INTEGER NOT NULL, plan TEXT NOT NULL, '
                     'next_index TEXT NOT NULL, PRIMARY KEY(iteration,round))')
        conn.execute('INSERT INTO iteration_combichem_history(iteration,round,plan,next_index) VALUES (?,?,?,?)',
                     (iteration, round_id, row[0], row[1]))
        conn.execute('UPDATE iteration_combichem SET plan=?,next_index=? WHERE iteration=?',
                     (json.dumps(following, sort_keys=True), '0', iteration))
    return dict(following, next_index=0)


def iteration_compound_count(path, iteration):
    with connect(path) as conn:
        return conn.execute('SELECT COUNT(*) FROM compound WHERE iteration=?', (iteration,)).fetchone()[0]


def elite_mcs_was_used(path, supplied):
    """Check durable generation snapshots, excluding the fixed docking row."""
    import json
    # Ensure the non-destructive scaffold migration exists for older projects.
    iteration_scaffold(path, 0)
    with connect(path) as conn:
        return any(json.loads(snapshot).get('supplied_elite_mcs') == supplied
                   for (snapshot,) in conn.execute(
                       'SELECT snapshot FROM iteration_scaffold WHERE iteration>0'))


def prior_generation_exists(path, iteration):
    """Recognize pre-marker generation history as well as current snapshots."""
    iteration_scaffold(path, 0)
    with connect(path) as conn:
        return conn.execute(
            'SELECT 1 FROM compound WHERE iteration>0 AND iteration<? '
            'UNION ALL SELECT 1 FROM iteration_scaffold WHERE iteration>0 AND iteration<? LIMIT 1',
            (iteration, iteration),
        ).fetchone() is not None


def fixed_generation_scaffold(path, iteration, supplied):
    """Reuse the first pinned scaffold, or the original supplied scaffold."""
    import json
    iteration_scaffold(path, 0)
    with connect(path) as conn:
        snapshots = [(i, json.loads(s)) for i, s in conn.execute(
            'SELECT iteration,snapshot FROM iteration_scaffold '
            'WHERE iteration>0 AND iteration<? ORDER BY iteration', (iteration,))]
    for i, snapshot in snapshots:
        if snapshot.get('elite_mcs_fixed') is True:
            return i, snapshot
    if supplied is not None:
        for i, snapshot in snapshots:
            if (snapshot.get('supplied_elite_mcs') == supplied
                    or (snapshot.get('source') == 'user_initial_mcs'
                        and snapshot.get('mcs_smarts') == supplied)):
                return i, snapshot
    return None


def missing_sa_scores(path: str | Path, iteration: int) -> list[tuple[int, str]]:
    with connect(path) as conn:
        return conn.execute(
            "SELECT id, smiles FROM compound "
            "WHERE iteration=? AND iteration>0 AND COALESCE(rdk_sa_score,sa_score) IS NULL ORDER BY id",
            (iteration,),
        ).fetchall()


def missing_rdkit_descriptors(path, iteration):
    missing = ' OR '.join(f'{name} IS NULL' for name in (*RDK_DESCRIPTOR_TYPES, 'sa_score'))
    with connect(path) as conn:
        return conn.execute(
            'SELECT id,smiles,COALESCE(rdk_sa_score,sa_score) FROM compound '
            f'WHERE iteration=? AND ({missing}) ORDER BY id', (iteration,),
        ).fetchall()


def novel_generated_smiles(path, smiles):
    with connect(path) as conn:
        existing = {_core.canonicalize_smiles(row[0]) for row in conn.execute('SELECT smiles FROM compound')}
    return [value for value in smiles if value not in existing]


def insert_generated_candidates(path, iteration, target_count, candidates, *, combichem_progress=None, raw=False):
    """Recheck duplicates and atomically cap the iteration at its total target."""
    if iteration <= 0:
        raise ValueError('Generated candidates require a positive iteration')
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        if raw and conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_druglike_selection'").fetchone():
            if conn.execute('SELECT 1 FROM iteration_druglike_selection WHERE iteration=?', (iteration,)).fetchone():
                _validate_druglike_population(conn, iteration)
                raise RuntimeError('Cannot append compounds to a frozen drug likeness population')
        if combichem_progress is not None:
            start, end, *round_ids = combichem_progress
            row = conn.execute('SELECT next_index,plan FROM iteration_combichem WHERE iteration=?',
                               (iteration,)).fetchone()
            if row is None:
                raise RuntimeError('Combichem batch has no saved plan')
            import json
            plan = json.loads(row[1])
            round_id = round_ids[0] if len(round_ids) == 1 else 0
            if len(round_ids) > 1 or round_id != plan.get('round', 0):
                raise RuntimeError('Stale or invalid combichem batch round')
            total = int(plan['total_combinations'])
            if not 0 <= start < end <= total:
                raise RuntimeError('Invalid combichem batch bounds')
            current = int(row[0])
            if current == end:
                return 0  # Replay after commit: never readmit rejected batch rows.
            if current != start or end <= start:
                raise RuntimeError('Stale or invalid combichem batch cursor')
        existing = {_core.canonicalize_smiles(row[0]) for row in conn.execute('SELECT smiles FROM compound')}
        names = {row[0] for row in conn.execute('SELECT name FROM compound')}
        if combichem_progress is not None:
            conn.execute('CREATE TABLE IF NOT EXISTS combichem_batch ('
                         'id INTEGER PRIMARY KEY AUTOINCREMENT, iteration INTEGER NOT NULL, '
                         'round INTEGER NOT NULL, start_index TEXT NOT NULL, end_index TEXT NOT NULL, '
                         'revision INTEGER NOT NULL, inserted INTEGER NOT NULL, UNIQUE(iteration,round,start_index))')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_combichem_batch_revision '
                         'ON combichem_batch(iteration,revision,id)')
            conn.execute('CREATE TABLE IF NOT EXISTS combichem_compound_origin ('
                         'compound_id INTEGER PRIMARY KEY REFERENCES compound(id) ON DELETE CASCADE, '
                         'iteration INTEGER NOT NULL, round INTEGER NOT NULL)')
        count = conn.execute('SELECT COUNT(*) FROM compound WHERE iteration=?', (iteration,)).fetchone()[0]
        remaining = max(0, target_count - count)
        inserted = sequence = 0
        columns = tuple(RDK_DESCRIPTOR_TYPES)
        for row in candidates:
            if inserted >= remaining:
                break
            smiles = row['smiles']
            if (not raw and not row['druglike_selected']) or smiles in existing:
                continue
            sequence += 1
            name = f'GEN_I{iteration:02d}_{sequence:06d}'
            while name in names:
                sequence += 1
                name = f'GEN_I{iteration:02d}_{sequence:06d}'
            values = [None if raw else row[column] for column in columns]
            if not raw and not all(math.isfinite(value) for value in values):
                raise ValueError('Nonfinite RDKit property in generated candidate')
            marks = ','.join('?' for _ in range(len(columns) + 5))
            cursor = conn.execute(
                f'INSERT INTO compound(name,smiles,iteration,state,sa_score,{",".join(columns)}) '
                f'VALUES ({marks})',
                [name, smiles, iteration, State.GENERATED, None if raw else row['rdk_sa_score'], *values],
            )
            if combichem_progress is not None:
                conn.execute('INSERT INTO combichem_compound_origin(compound_id,iteration,round) VALUES (?,?,?)',
                             (cursor.lastrowid, iteration, round_id))
            existing.add(smiles)
            names.add(name)
            inserted += 1
        if combichem_progress is not None:
            conn.execute('UPDATE iteration_combichem SET next_index=? WHERE iteration=?',
                         (str(end), iteration))
            conn.execute('INSERT INTO combichem_batch(iteration,round,start_index,end_index,revision,inserted) '
                         'VALUES (?,?,?,?,?,?)',
                         (iteration, round_id, str(start), str(end), plan.get('attachment_revision', 0), inserted))
        return inserted


def store_rdkit_descriptors(path, results):
    """Commit a complete property batch, filling only missing values."""
    columns = tuple(RDK_DESCRIPTOR_TYPES)
    assignments = ','.join(
        'rdk_sa_score=COALESCE(rdk_sa_score,sa_score,?)' if name == 'rdk_sa_score'
        else f'{name}=COALESCE({name},?)' for name in columns)
    missing = ' OR '.join(f'{name} IS NULL' for name in (*columns, 'sa_score'))
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        updated = 0
        for cid, values in results:
            if set(values) != set(columns) or not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f'Incomplete or nonfinite RDKit properties for compound {cid}')
            updated += conn.execute(
                f'UPDATE compound SET {assignments},sa_score=COALESCE(sa_score,rdk_sa_score,?),'
                f'modified_at=CURRENT_TIMESTAMP WHERE id=? AND ({missing})',
                [*(values[name] for name in columns), values['rdk_sa_score'], cid],
            ).rowcount
        return updated


def apply_sa_filter(path, iteration, target_count, ratio):
    """Freeze final SA selection atomically, before any downstream work starts.

    Return a saved selection, or None for underfilled/legacy started iterations.
    Never modify running work, existing scientific results or failure records.
    """
    from molnova.config import sa_accepted_ratio
    ratio = sa_accepted_ratio({'sa-accepted-ratio': ratio})
    if iteration <= 0:
        return None
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        if _uses_raw_generation(conn, iteration):
            return None  # New iterations apply optional SA selection within drug likeness.
        conn.execute(
            'CREATE TABLE IF NOT EXISTS iteration_sa_selection ('
            'iteration INTEGER PRIMARY KEY, ratio REAL NOT NULL, '
            'compound_count INTEGER NOT NULL, accepted_count INTEGER NOT NULL)'
        )
        saved = conn.execute(
            'SELECT ratio,compound_count,accepted_count FROM iteration_sa_selection WHERE iteration=?',
            (iteration,),
        ).fetchone()
        if saved:
            return saved
        rows = conn.execute(
            'SELECT id,COALESCE(rdk_sa_score,sa_score),state,docking_score,amber_gbsa_score,gbsa_score,fep_score '
            'FROM compound WHERE iteration=? ORDER BY COALESCE(rdk_sa_score,sa_score),id', (iteration,),
        ).fetchall()
        if len(rows) < target_count:
            return None
        if any(state != State.GENERATED or any(score is not None for score in results)
               for _, _, state, *results in rows):
            return None
        if any(score is None or not math.isfinite(score) or not 1 <= score <= 10
               for _, score, *_ in rows):
            raise RuntimeError(f'Iteration {iteration}: SA selection requires all valid SA scores')
        accepted = math.ceil(len(rows) * ratio)
        conn.executemany(
            'UPDATE compound SET state=?,modified_at=CURRENT_TIMESTAMP WHERE id=? AND state=?',
            [(State.SA_REJECTED, cid, State.GENERATED) for cid, *_ in rows[accepted:]],
        )
        conn.execute(
            'INSERT INTO iteration_sa_selection VALUES (?,?,?,?)',
            (iteration, ratio, len(rows), accepted),
        )
        return ratio, len(rows), accepted


def ligprep_pending_rows(path, iteration, synthetic_enabled):
    """Include only the known missing-input Glide failure for preparation retry."""
    with connect(path) as conn:
        return conn.execute(
            "SELECT id,name,smiles FROM compound WHERE iteration=? "
            "AND (state IN (?,?) OR (state=? AND failed_stage='glide' "
            "AND failure_message='No LigPrep structures for reference group' "
            "AND docking_score IS NULL AND gbsa_score IS NULL AND fep_score IS NULL)) "
            "AND (?=0 OR synthetic_feasibility IS NOT NULL) ORDER BY id",
            (iteration, State.GENERATED, State.LIGPREP_RUNNING, State.FAILED, int(synthetic_enabled)),
        ).fetchall()


def fresh_glide_references(path, iteration):
    with connect(path) as conn:
        return {row[0] for row in conn.execute(
            'SELECT DISTINCT similar_to FROM compound WHERE iteration=? '
            'AND state=? AND docking_score IS NULL AND similar_to IS NOT NULL',
            (iteration, State.LIGPREPPED),
        )}


def store_sa_scores(path: str | Path, scores: Iterable[tuple[int, float]]) -> int:
    """Persist a score batch atomically, preserving existing results and states."""
    with connect(path) as conn:
        updated = 0
        for compound_id, score in scores:
            updated += conn.execute(
                "UPDATE compound SET rdk_sa_score=COALESCE(rdk_sa_score,sa_score,?), "
                "sa_score=COALESCE(sa_score,rdk_sa_score,?), modified_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND iteration>0 AND (sa_score IS NULL OR rdk_sa_score IS NULL)",
                (score, score, compound_id),
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
                "UPDATE compound SET rdk_sa_score=COALESCE(rdk_sa_score,sa_score,?), "
                "sa_score=COALESCE(sa_score,rdk_sa_score,?), modified_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND iteration=? AND (sa_score IS NULL OR rdk_sa_score IS NULL)",
                (score, score, compound_id, iteration),
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


def mmgbsa_progress_counts(path, iteration):
    """Read iteration-wide running claims and finite persisted GBSA results."""
    with connect(path) as conn:
        rows = conn.execute('SELECT state,gbsa_score FROM compound WHERE iteration=?',
                            (iteration,)).fetchall()
    running = sum(state == State.GBSA_RUNNING for state, _ in rows)
    saved = sum(score is not None and math.isfinite(score) for _, score in rows)
    return running, saved


def mmgbsa_scored_ids(path, iteration):
    """IDs whose finite Prime scores may be paired with native coordinates."""
    with connect(path) as conn:
        rows = conn.execute('SELECT id,gbsa_score FROM compound WHERE iteration=? ORDER BY id',
                            (iteration,)).fetchall()
    return [cid for cid, score in rows if score is not None and math.isfinite(score)]


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
    if stage == 'druglike':
        return druglike_completion_reason(path, iteration)
    if stage in {'synthetic', 'ligprep', 'glide'}:
        reason = druglike_completion_reason(path, iteration)
        if reason:
            return reason
    with connect(path) as conn:
        rows = _iteration_rows(conn, iteration)
        if len(rows) < target_count:
            return f'generation incomplete ({len(rows)}/{target_count} compounds)'
        if stage == 'generate':
            return None
        if stage == 'synthetic':
            pending = conn.execute(
                'SELECT COUNT(*) FROM compound WHERE iteration=? AND state NOT IN (?,?,?) '
                'AND (synthetic_feasibility IS NULL OR sa_score IS NULL OR state=?)',
                (iteration, State.FAILED, State.SA_REJECTED, State.DRUGLIKE_REJECTED, State.SYNTHETIC_RUNNING),
            ).fetchone()[0]
            return f'{pending} compounds await synthetic assessment' if pending else None
        if stage == 'ligprep':
            blocked = {State.GENERATED, State.SYNTHETIC_RUNNING, State.LIGPREP_RUNNING}
            pending = sum(state in blocked for _, state, _, _ in rows)
            if pending:
                return f'{pending} compounds await LigPrep completion'
            if synthetic_enabled and conn.execute(
                'SELECT 1 FROM compound WHERE iteration=? AND state NOT IN (?,?,?) '
                'AND synthetic_feasibility IS NULL LIMIT 1', (iteration, State.FAILED, State.SA_REJECTED, State.DRUGLIKE_REJECTED)
            ).fetchone():
                return 'synthetic assessment incomplete'
            return None
        if stage == 'glide':
            return _docking_completion_reason(rows, target_count)
    raise ValueError(f'Unknown stage: {stage}')



def stage_summary_results(path, iteration, stage):
    """Read result identities for display only; never claim or change work."""
    with connect(path) as conn:
        rows = conn.execute(
            'SELECT id,state,failed_stage,docking_score,amber_gbsa_score,gbsa_score '
            'FROM compound WHERE iteration=?', (iteration,),
        ).fetchall()
    failure_stage = {'mmgbsa': 'gbsa', 'amber-gbsa': 'amber_gbsa'}.get(stage, stage)
    failures = {cid for cid, state, failed, *_ in rows
                if failed == failure_stage and (state == State.FAILED or stage == 'amber-gbsa')}
    if stage == 'generate':
        successes = {row[0] for row in rows if row[1] not in {State.SA_REJECTED, State.DRUGLIKE_REJECTED}}
        if uses_raw_generation(path, iteration):
            successes = {row[0] for row in rows}
    elif stage == 'ligprep':
        advanced = {State.LIGPREPPED, State.GLIDE_RUNNING, State.DOCKED,
                    State.AMBER_RUNNING, State.GBSA_RUNNING, State.GBSA_DONE,
                    State.FEP_RUNNING, State.FEP_DONE}
        successes = {cid for cid, state, failed, *_ in rows
                     if state in advanced or
                     (state == State.FAILED and failed in {'glide', 'amber_gbsa', 'gbsa', 'fep'})}
    else:
        column = {'glide': 3, 'amber-gbsa': 4, 'mmgbsa': 5}[stage]
        successes = {row[0] for row in rows
                     if row[column] is not None and math.isfinite(row[column])}
    return successes, failures - successes


def start_stage_timer(path, iteration, stage):
    """Keep the first supervised stage start across polls and driver restarts."""
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(
            'CREATE TABLE IF NOT EXISTS stage_timing ('
            'iteration INTEGER NOT NULL, stage TEXT NOT NULL, started_at REAL NOT NULL, '
            'finished_at REAL, PRIMARY KEY (iteration, stage))'
        )
        conn.execute(
            'INSERT INTO stage_timing(iteration,stage,started_at) VALUES (?,?,?) '
            'ON CONFLICT(iteration,stage) DO NOTHING', (iteration, stage, time.time()),
        )


def finish_stage_timer(path, iteration, stage):
    """Record verified full-stage completion once, including wait/restart downtime."""
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(
            'UPDATE stage_timing SET finished_at=COALESCE(finished_at, ?) '
            'WHERE iteration=? AND stage=?', (time.time(), iteration, stage),
        )
        row = conn.execute(
            'SELECT started_at,finished_at FROM stage_timing WHERE iteration=? AND stage=?',
            (iteration, stage),
        ).fetchone()
        if row is None:
            raise RuntimeError(f'Iteration {iteration} stage {stage} has no start time')
        return max(0, row[1] - row[0])


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


def fep_candidate_iterations(path):
    with connect(path) as conn:
        return [row[0] for row in conn.execute(
            'SELECT DISTINCT iteration FROM compound WHERE iteration>0 AND '
            '(gbsa_score IS NOT NULL OR amber_gbsa_score IS NOT NULL OR docking_score IS NOT NULL) ORDER BY iteration')]


def fep_selection_pending(path, iteration, count):
    import json
    with connect(path) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_fep_selection'").fetchone():
            saved = conn.execute('SELECT candidates FROM iteration_fep_selection WHERE iteration=?', (iteration,)).fetchone()
            if saved:
                ids = {item['id'] for item in json.loads(saved[0])}
                return any(cid in ids and score is None for cid, score in conn.execute(
                    'SELECT id,fep_score FROM compound WHERE iteration=?', (iteration,)))
        for column in ('gbsa_score', 'amber_gbsa_score', 'docking_score'):
            rows = conn.execute(f'SELECT id,{column},fep_score FROM compound WHERE iteration=?', (iteration,)).fetchall()
            ranked = sorted((row for row in rows if row[1] is not None and math.isfinite(row[1])),
                            key=lambda row: (row[1], row[0]))
            if ranked:
                return any(row[2] is None for row in ranked[:count])
    return False


def fep_candidates(path, iteration, count, *, target_count=1, prime_count=None,
                   amber_keep_count=None, for_import=False):
    """Freeze one score source and candidate list; never compare unlike energies."""
    import json
    if iteration <= 0:
        raise ValueError('FEP candidate selection requires iteration > 0')
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('CREATE TABLE IF NOT EXISTS iteration_fep_selection ('
                     'iteration INTEGER PRIMARY KEY,ranking_source TEXT NOT NULL,candidates TEXT NOT NULL)')
        saved = conn.execute('SELECT ranking_source,candidates FROM iteration_fep_selection WHERE iteration=?',
                             (iteration,)).fetchone()
        rows = conn.execute('SELECT id,name,smiles,docking_score,amber_gbsa_score,gbsa_score,fep_score,state '
                            'FROM compound WHERE iteration=? ORDER BY id', (iteration,)).fetchall()
        if saved:
            source, selected = saved[0], json.loads(saved[1])
        else:
            if for_import and _uses_raw_generation(conn, iteration):
                return []  # Imports cannot implicitly select an unexported raw iteration.
            if not for_import:
                reason = _docking_completion_reason(_iteration_rows(conn, iteration), target_count)
                if reason:
                    raise RuntimeError(f'FEP waiting: {reason}')
                if _uses_raw_generation(conn, iteration):
                    if (not conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_druglike_selection'").fetchone()
                            or not conn.execute('SELECT 1 FROM iteration_druglike_selection WHERE iteration=?', (iteration,)).fetchone()):
                        raise RuntimeError('FEP waiting for drug likeness')
                if any(row[7] in {State.AMBER_RUNNING, State.GBSA_RUNNING} for row in rows):
                    raise RuntimeError('FEP waiting for active Amber/Prime claims')
                if amber_keep_count is not None:
                    reason = _amber_completion_reason(conn, iteration, target_count, amber_keep_count)
                    if reason:
                        raise RuntimeError(f'FEP waiting: {reason}')
                if prime_count is not None and _mmgbsa_candidates(conn, iteration, prime_count,
                                                                target_count, amber_keep_count):
                    raise RuntimeError('FEP waiting for unfinished Prime candidates')
            selected, source = [], 'docking'
            sources = [('prime', 5)] if for_import else [('prime', 5), ('amber', 4), ('docking', 3)]
            for name, column in sources:
                eligible = [row for row in rows if row[column] is not None and math.isfinite(row[column])]
                if eligible:
                    eligible.sort(key=lambda row: (row[column], row[0]))
                    source = name
                    selected = [dict(id=row[0], ranking_score=row[column]) for row in eligible[:count]]
                    break
            conn.execute('INSERT INTO iteration_fep_selection VALUES (?,?,?)',
                         (iteration, source, json.dumps(selected)))
        by_id = {row[0]: row for row in rows}
        fields = ['id', 'name', 'smiles', 'docking_score', 'amber_gbsa_score', 'gbsa_score', 'fep_score', 'state']
        result = []
        for item in selected:
            if item['id'] not in by_id:
                raise RuntimeError('Frozen FEP candidate is missing from its iteration')
            result.append(dict(zip(fields, by_id[item['id']]), ranking_source=source,
                               ranking_score=item['ranking_score']))
        return result


def store_fep_scores(path, updates, allowed_ids):
    """Validate and import the entire batch without changing other scientific scores."""
    with connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        if set(updates) - set(allowed_ids):
            raise ValueError('FEP scores include IDs outside the selected candidate set')
        for cid, score in updates.items():
            if not math.isfinite(score):
                raise ValueError(f'FEP score for {cid} must be finite')
            row = conn.execute('SELECT fep_score FROM compound WHERE id=? AND iteration>0', (cid,)).fetchone()
            if row is None:
                raise ValueError(f'Unknown FEP candidate {cid}')
            if row[0] is not None and row[0] != score:
                raise ValueError(f'Conflicting existing FEP score for {cid}')
        for cid, score in updates.items():
            conn.execute("UPDATE compound SET fep_score=?,state=?,modified_at=CURRENT_TIMESTAMP "
                         'WHERE id=? AND fep_score IS NULL', (score, State.FEP_DONE, cid))
        return len(updates)
