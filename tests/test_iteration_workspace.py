import json
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
    (folder / 'glide_constrained.log').write_text('completed calculation')
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
    ('prime_job.json', '{"mode":"compound_queue_v1"}'),
    ('ligprep_job.json', '{"job_id":null,"launching":true}'),
    ('glide_job.tmp', '{"job_id":null}'),
    ('amber_job.json', '{}'),
    ('sampling.remote-dir.txt', 'tensor:/old/run'),
    ('ligprep_all.log', 'uncertain legacy calculation'),
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
    (folder / 'glide_constrained.log').write_text('JobId: legacy\n')
    checked = []
    def details(suite, job_id, directory):
        checked.append((job_id, directory))
        return [{'jobId': job_id, 'status': 'DONE'}]
    monkeypatch.setattr(guard, 'job_details', details)
    assert prepare_iteration_directory(workspace, 1)
    assert checked == [('legacy', folder)]
