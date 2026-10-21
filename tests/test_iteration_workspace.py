import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, schrodinger_guard as guard
from molnova.stages._workspace import prepare_iteration_directory


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('reference','C',0,'reference')")
    root = tmp_path / 'iter1'
    (root / 'glide/ref_1').mkdir(parents=True)
    (root / 'glide/ref_1/best_poses.maegz').write_text('old poses')
    (root / 'combichem').mkdir()
    (root / 'combichem/plan.json').write_text('old plan')
    monkeypatch.setattr(guard, 'active_jobs', lambda *a: [])
    return SimpleNamespace(db_path=db, output=tmp_path, schrodinger=Path('/suite'))


def test_new_iteration_clears_all_old_subdirectories_before_generation(workspace):
    database.start_stage_timer(workspace.db_path, 1, 'generate')
    with database.connect(workspace.db_path) as conn:
        before = conn.execute('SELECT * FROM compound').fetchall()
    archive = prepare_iteration_directory(workspace, 1)
    assert list((workspace.output / 'iter1').iterdir()) == []
    assert archive.parent == workspace.output / '.previous'
    assert (archive / 'glide/ref_1/best_poses.maegz').read_text() == 'old poses'
    assert (archive / 'combichem/plan.json').read_text() == 'old plan'
    with database.connect(workspace.db_path) as conn:
        assert conn.execute('SELECT * FROM compound').fetchall() == before
        assert conn.execute('SELECT iteration FROM iteration_workspace').fetchall() == [(1,)]
    assert prepare_iteration_directory(workspace, 1) is None


def test_checkpoint_protects_zero_row_retry_before_scaffold_is_saved(workspace, monkeypatch):
    prepare_iteration_directory(workspace, 1)
    sample = workspace.output / 'iter1/generated.csv'
    sample.write_text('SMILES\nCCO\n')
    monkeypatch.setattr(guard, 'active_jobs', lambda *a: pytest.fail('cleaned a resumed workspace'))
    assert prepare_iteration_directory(workspace, 1) is None
    assert sample.read_text() == 'SMILES\nCCO\n'


@pytest.mark.parametrize('saved', ['compound', 'scaffold', 'plan'])
def test_resuming_saved_iteration_never_clears_its_files(workspace, monkeypatch, saved):
    if saved == 'compound':
        with database.connect(workspace.db_path) as conn:
            conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('generated','CC',1,'generated')")
    elif saved == 'scaffold':
        database.iteration_scaffold(workspace.db_path, 1, {'scaffold_smiles': 'C[*:0]'})
    else:
        database.combichem_plan(workspace.db_path, 1, {'total_combinations': '1'})
    monkeypatch.setattr(guard, 'active_jobs', lambda *a: pytest.fail('queried jobs on resume'))
    assert prepare_iteration_directory(workspace, 1) is None
    assert (workspace.output / 'iter1/combichem/plan.json').read_text() == 'old plan'
    assert not (workspace.output / '.previous').exists()


@pytest.mark.parametrize('status', ['RUNNING', 'UNKNOWN', 'DONE', 'lookup_failed'])
def test_cleanup_checks_saved_native_parent_and_children_before_moving_anything(workspace, monkeypatch, status):
    folder = workspace.output / 'iter1/glide/ref_1'
    (folder / 'glide_job.json').write_text(json.dumps({'job_id': 'old'}))
    (folder / 'glide.log').write_text('completed calculation')
    def details(*a):
        if status == 'lookup_failed':
            raise OSError('JobServer unavailable')
        return [{'jobId': 'old', 'status': 'DONE'}, {'jobId': 'child', 'status': status}]
    monkeypatch.setattr(guard, 'job_details', details)
    if status == 'DONE':
        assert prepare_iteration_directory(workspace, 1) is not None
    else:
        with pytest.raises((RuntimeError, OSError)):
            prepare_iteration_directory(workspace, 1)
        assert (folder / 'best_poses.maegz').read_text() == 'old poses'
        assert not (workspace.output / '.previous').exists()


@pytest.mark.parametrize('empty', [False, True])
def test_live_job_without_local_record_blocks_cleanup(workspace, monkeypatch, empty):
    launch = workspace.output / ('iter1/glide/ref_empty' if empty else 'iter1/glide/ref_1')
    launch.mkdir(exist_ok=True)
    monkeypatch.setattr(guard, 'active_jobs', lambda *a: [
        {'jobId': 'live', 'spec': {'launchParams': {'launchDirectory': str(launch)}}}])
    with pytest.raises(RuntimeError, match='live external jobs'):
        prepare_iteration_directory(workspace, 1)
    assert (workspace.output / 'iter1/glide/ref_1/best_poses.maegz').exists()


def test_jobserver_lookup_failure_retains_every_file(workspace, monkeypatch):
    def offline(*a):
        raise OSError('JobServer unavailable')
    monkeypatch.setattr(guard, 'active_jobs', offline)
    with pytest.raises(OSError, match='JobServer unavailable'):
        prepare_iteration_directory(workspace, 1)
    assert (workspace.output / 'iter1/glide/ref_1/best_poses.maegz').exists()
    assert not (workspace.output / '.previous').exists()
    assert not database.iteration_has_saved_work(workspace.db_path, 1)


@pytest.mark.parametrize('filename, contents', [
    ('prime_gbsa_job.json', '{"mode":"compound_queue_v1"}'),
    ('ligprep_job.json', '{"job_id":null,"launching":true}'),
    ('glide_job.tmp', '{"job_id":null}'),
    ('amber_job.json', '{}'),
    ('sampling.remote-dir.txt', 'tensor:/old/run'),
    ('ligprep.log', 'uncertain legacy calculation'),
])
def test_unresolved_submissions_preserve_entire_iteration(workspace, filename, contents):
    root = workspace.output / 'iter1'
    (root / filename).write_text(contents)
    with pytest.raises(RuntimeError, match='retained'):
        prepare_iteration_directory(workspace, 1)
    assert (root / filename).read_text() == contents
    assert not (workspace.output / '.previous').exists()


def test_cleanup_preserves_configuration_inputs_and_iteration_zero(workspace):
    workspace.sidechain_library = workspace.output / 'iter1/edges.smi'
    workspace.sidechain_library.write_text('*C\n')
    with pytest.raises(RuntimeError, match='move the input'):
        prepare_iteration_directory(workspace, 1)
    assert workspace.sidechain_library.read_text() == '*C\n'
    with pytest.raises(ValueError, match='iteration >= 1'):
        prepare_iteration_directory(workspace, 0)


def test_legacy_jobid_in_log_is_checked_before_cleanup(workspace, monkeypatch):
    folder = workspace.output / 'iter1/glide/ref_1'
    (folder / 'glide.log').write_text('JobId: legacy\n')
    checked = []
    def details(suite, job_id, directory):
        checked.append((job_id, directory))
        return [{'jobId': job_id, 'status': 'DONE'}]
    monkeypatch.setattr(guard, 'job_details', details)
    assert prepare_iteration_directory(workspace, 1)
    assert checked == [('legacy', folder)]


@pytest.fixture
def deleted(workspace):
    database.mark_iteration_workspace_prepared(workspace.db_path, 1)
    database.insert_generated_candidates(workspace.db_path, 1, 2,
                                         [{'smiles': 'CC'}, {'smiles': 'CCC'}], raw=True)
    population = database.druglike_population(workspace.db_path, 1)
    database.apply_druglike_selection(workspace.db_path, 1, 2, [
        dict(id=row['id'], sa_score=1, druglike_selected=True, druglike_reject_reason=None)
        for row in population])
    database.combichem_plan(workspace.db_path, 1, {'total_combinations': '3'})
    with database.connect(workspace.db_path) as conn:
        conn.execute('CREATE TABLE combichem_compound_origin ('
                     'compound_id INTEGER PRIMARY KEY REFERENCES compound(id) ON DELETE CASCADE, '
                     'iteration INTEGER NOT NULL, round INTEGER NOT NULL)')
        conn.executemany('INSERT INTO combichem_compound_origin VALUES (?,1,0)',
                         [(row['id'],) for row in population])
        conn.execute("UPDATE iteration_combichem SET next_index='3' WHERE iteration=1")
    # Reproduce manual deletion using a SQLite client without FK enforcement.
    with sqlite3.connect(workspace.db_path) as conn:
        conn.execute('DELETE FROM compound WHERE iteration=1')
    return workspace


def test_deleted_population_backs_up_wal_and_resets_all_metadata_before_generation(deleted):
    assert database.iteration_was_deleted(deleted.db_path, 1)
    archive = prepare_iteration_directory(deleted, 1)
    assert (archive / 'combichem/plan.json').read_text() == 'old plan'
    backup = next(archive.glob('database-before-reset-*.sqlite'))
    with sqlite3.connect(backup) as conn:
        assert conn.execute('SELECT population_count FROM iteration_druglike_selection').fetchall() == [(2,)]
        assert conn.execute('SELECT COUNT(*) FROM compound_druglike_selection').fetchone() == (2,)
        assert conn.execute('SELECT next_index FROM iteration_combichem').fetchone() == ('3',)
    with database.connect(deleted.db_path) as conn:
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
        assert conn.execute('SELECT COUNT(*) FROM compound').fetchone() == (1,)
        for table in ('iteration_druglike_selection', 'compound_druglike_selection',
                      'combichem_compound_origin', 'iteration_combichem'):
            assert conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone() == (0,)
    assert database.uses_raw_generation(deleted.db_path, 1)
    assert not database.iteration_was_deleted(deleted.db_path, 1)
    assert list((deleted.output / 'iter1').iterdir()) == []
    # A second poll resumes the fresh workspace without another backup/reset.
    assert prepare_iteration_directory(deleted, 1) is None


@pytest.mark.parametrize('status', ['RUNNING', 'UNKNOWN', 'lookup_failed'])
def test_deleted_population_is_not_reset_until_native_work_is_verified_terminal(deleted, monkeypatch, status):
    folder = deleted.output / 'iter1/glide/ref_1'
    (folder / 'glide_job.json').write_text(json.dumps({'job_id': 'old'}))
    def details(*a):
        if status == 'lookup_failed':
            raise OSError('unavailable')
        return [{'jobId': 'old', 'status': 'DONE'}, {'jobId': 'child', 'status': status}]
    monkeypatch.setattr(guard, 'job_details', details)
    with pytest.raises((RuntimeError, OSError)):
        prepare_iteration_directory(deleted, 1)
    assert database.iteration_was_deleted(deleted.db_path, 1)
    assert (folder / 'best_poses.maegz').is_file()
    assert not (deleted.output / '.previous').exists()


def test_reset_refuses_another_active_stage(deleted):
    with database.stage_lock(deleted.db_path, 'glide'):
        with pytest.raises(RuntimeError, match='already running'):
            prepare_iteration_directory(deleted, 1)
    assert database.iteration_was_deleted(deleted.db_path, 1)
    assert not (deleted.output / '.previous').exists()


def test_reset_rolls_back_and_restores_files_then_can_retry(deleted, monkeypatch):
    original = database._mark_iteration_workspace_prepared
    def fail(*a):
        raise OSError('interrupted metadata reset')
    monkeypatch.setattr(database, '_mark_iteration_workspace_prepared', fail)
    with pytest.raises(OSError):
        prepare_iteration_directory(deleted, 1)
    assert database.iteration_was_deleted(deleted.db_path, 1)
    assert (deleted.output / 'iter1/combichem/plan.json').read_text() == 'old plan'
    monkeypatch.setattr(database, '_mark_iteration_workspace_prepared', original)
    archive = prepare_iteration_directory(deleted, 1)
    assert len(list(archive.glob('database-before-reset-*.sqlite'))) == 2
    assert database.druglike_completion_reason(deleted.db_path, 1) is not None


def test_fresh_generation_clears_deleted_population_before_sampling(deleted, monkeypatch):
    from molnova.stages import generate
    deleted.target_count = 2
    deleted.max_iteration = 1
    deleted.sidechain_library = deleted.output / 'edges.smi'
    monkeypatch.setattr(generate, 'prepare_iteration_scaffold', lambda *a, **kw: None)
    calls = []
    def sample(args, iteration, directory):
        assert database.druglike_completion_reason(args.db_path, iteration) is not None
        assert database.combichem_plan(args.db_path, iteration) is None
        calls.append(iteration)
    monkeypatch.setattr(generate, '_run_combichem', sample)
    generate._run_stage(deleted, 1)
    assert calls == [1]


def test_partial_manual_deletion_fails_before_sampling_without_reset(deleted, monkeypatch):
    from molnova.stages import generate
    with database.connect(deleted.db_path) as conn:
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('partial','CC',1,'generated')")
    deleted.target_count = 2
    deleted.max_iteration = 1
    monkeypatch.setattr(generate, '_run_combichem', lambda *a: pytest.fail('sampled corrupt population'))
    with pytest.raises(RuntimeError, match='frozen drug likeness population is inconsistent'):
        generate._run_stage(deleted, 1)
    assert not (deleted.output / '.previous').exists()


@pytest.mark.parametrize('ownership', ['report', 'missing', 'disagree'])
def test_reset_requires_proven_original_membership(deleted, ownership):
    with database.connect(deleted.db_path) as conn:
        ids = [row[0] for row in conn.execute('SELECT compound_id FROM combichem_compound_origin')]
        if ownership != 'disagree':
            conn.execute('DELETE FROM combichem_compound_origin')
    if ownership != 'missing':
        report = deleted.output / 'iter1/druglike/druglike_selection.csv'
        report.parent.mkdir()
        members = ids if ownership == 'report' else ids[:1]
        report.write_text('id\n' + ''.join(f'{cid}\n' for cid in members))
    if ownership == 'report':
        prepare_iteration_directory(deleted, 1)
        with database.connect(deleted.db_path) as conn:
            assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
    else:
        with pytest.raises(RuntimeError, match='required before reset|disagree'):
            prepare_iteration_directory(deleted, 1)
        assert database.iteration_was_deleted(deleted.db_path, 1)
        assert (deleted.output / 'iter1/combichem/plan.json').read_text() == 'old plan'


def test_backup_contains_committed_wal_frames(deleted):
    # Keep a reader snapshot open so the subsequent commit cannot be folded
    # into the main database file before the backup.
    with sqlite3.connect(deleted.db_path) as reader:
        reader.execute('BEGIN')
        reader.execute('SELECT * FROM iteration_combichem').fetchall()
        with database.connect(deleted.db_path) as writer:
            writer.execute("UPDATE iteration_combichem SET next_index='2'")
        archive = prepare_iteration_directory(deleted, 1)
        with sqlite3.connect(next(archive.glob('database-before-reset-*.sqlite'))) as backup:
            assert backup.execute('SELECT next_index FROM iteration_combichem').fetchone() == ('2',)
