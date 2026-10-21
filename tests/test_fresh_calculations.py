import json
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, ligprep_recovery, primg_gbsa_async, primg_gbsa_queue, schrodinger_guard
from molnova.glide_recovery import group_status
from molnova.stages import ligprep, glide


@pytest.mark.parametrize('status', ['DONE', 'FAILED'])
def test_old_ligprep_files_are_cleared_before_new_input(tmp_path, monkeypatch, status):
    folder = tmp_path / 'ligprep'
    folder.mkdir()
    (folder / 'ligprep.log').write_text('JobId: old\nCMPID_30025\n')
    (folder / 'ligprep.maegz').write_text('old molecules')
    (folder / 'input_all.smi').write_text('C\tCMPID_30025\n')
    scratch = folder / 'native-scratch'
    scratch.mkdir()
    (scratch / 'partial').write_text('old intermediate')
    monkeypatch.setattr(schrodinger_guard, 'active_jobs', lambda *a: [])
    monkeypatch.setattr(schrodinger_guard, 'job_details', lambda *a: [{'jobId': 'old', 'status': status}])
    assert ligprep_recovery.prepare_new_calculation('/suite', folder, [40025]) is None
    assert list(folder.iterdir()) == []
    assert len(list(tmp_path.glob('ligprep.previous-*/native-scratch/partial'))) == 1


@pytest.mark.parametrize('status', ['RUNNING', 'UNKNOWN'])
def test_old_ligprep_live_or_unknown_job_preserves_all_files(tmp_path, monkeypatch, status):
    (tmp_path / 'ligprep.log').write_text('JobId: old\n')
    output = tmp_path / 'ligprep.maegz'
    output.write_text('preserved')
    monkeypatch.setattr(schrodinger_guard, 'active_jobs', lambda *a: [])
    monkeypatch.setattr(schrodinger_guard, 'job_details', lambda *a: [{'jobId': 'old', 'status': status}])
    with pytest.raises(RuntimeError, match='still running'):
        ligprep_recovery.prepare_new_calculation('/suite', tmp_path, [40025])
    assert output.read_text() == 'preserved'


def test_matching_ligprep_saved_submission_is_recovered_without_cleanup(tmp_path, monkeypatch):
    ligprep_recovery._save(tmp_path, {'job_id': 'same', 'compound_ids': [1]})
    output = tmp_path / 'ligprep.maegz'
    output.write_text('current')
    monkeypatch.setattr(schrodinger_guard, 'active_jobs', lambda *a: pytest.fail('cleaned current submission'))
    assert ligprep_recovery.prepare_new_calculation('/suite', tmp_path, [1])['job_id'] == 'same'
    assert output.read_text() == 'current'


def test_stage_retries_only_known_missing_ligprep_failure(tmp_path, monkeypatch):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany(
            "INSERT INTO compound(name,smiles,iteration,state,failed_stage,failure_message,sa_score) "
            "VALUES (?, 'CCO', 2, 'failed', 'glide', ?, 2.0)",
            [('retry', 'No LigPrep structures for reference group'), ('keep', 'No Glide pose')],
        )
    args = SimpleNamespace(db_path=db, output=tmp_path, schrodinger=Path('/suite'), target_count=2,
                           synthetic_feasibility_enabled=False, host='compute:128',
                           ligprep_ph=7.4, ligprep_pht=1.0, ligprep_max_states=2,
                           ligprep_max_stereo=1, ligprep_ring_confs=1)
    folder = tmp_path / 'iter2/ligprep'
    folder.mkdir(parents=True)
    (folder / 'ligprep.log').write_text('JobId: old\n')
    (folder / 'ligprep.maegz').write_text('old')
    monkeypatch.setattr(schrodinger_guard, 'active_jobs', lambda *a: [])
    monkeypatch.setattr(schrodinger_guard, 'job_details', lambda *a: [{'jobId': 'old', 'status': 'DONE'}])
    def calculate(**kw):
        assert kw['compound_ids'] == [1]
        assert kw['input_file'].read_text() == 'CCO\tCMPID_1\n'
        out = kw['glide_root'] / 'ligprep.maegz'
        assert not out.exists()
        out.write_text('fresh')
        return out
    monkeypatch.setattr(_core, 'run_ligprep_once', calculate)
    ligprep._run_stage(args, 2)
    with database.connect(db) as conn:
        assert conn.execute('SELECT state,sa_score FROM compound ORDER BY id').fetchall() == [
            ('ligprepped', 2.0), ('failed', 2.0)]


def test_glide_cached_output_without_submission_is_not_completion(tmp_path):
    output = tmp_path / 'glide_lib.maegz'
    output.write_text('cached')
    assert group_status('/suite', tmp_path, tmp_path / 'glide.log', output, []) is None


@pytest.mark.parametrize('status', ['RUNNING', 'UNKNOWN', 'DONE'])
def test_obsolete_queue_checks_each_native_child_before_cleanup(tmp_path, monkeypatch, status):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
    root = tmp_path / 'primg_gbsa'
    root.mkdir()
    record = {'mode': primg_gbsa_queue.MODE, 'compound_ids': [30025],
              'run_dir': 'queue-1', 'entries': {'30025': 'active'}}
    primg_gbsa_async._save(root, record)
    child = root / 'queue-1/CMPID_30025'
    child.mkdir(parents=True)
    primg_gbsa_async._save(child, {'command': ['/suite/prime_mmgbsa'], 'job_id': 'old', 'compound_ids': [30025]})
    monkeypatch.setattr(primg_gbsa_async, '_job_details', lambda *a: [
        {'jobId': 'old', 'status': 'DONE'}, {'jobId': 'child', 'status': status}])
    if status != 'DONE':
        with pytest.raises(primg_gbsa_async.PrimeReconciliationPending, match='live jobs'):
            primg_gbsa_async.reconcile_saved_job(root, db, 2)
        assert primg_gbsa_async.load_job(root) == record
    else:
        assert primg_gbsa_async.reconcile_saved_job(root, db, 2) is None
        assert list(root.iterdir()) == []
        assert len(list(tmp_path.glob('primg_gbsa.stale-*'))) == 1


def test_new_primg_gbsa_inputs_clear_working_files_but_preserve_scored_history(tmp_path, monkeypatch):
    monkeypatch.setattr(schrodinger_guard, 'active_jobs', lambda *a: [])
    (tmp_path / 'receptor.maegz').write_text('old receptor')
    (tmp_path / 'primg_gbsa_input.maegz').write_text('old PV')
    (tmp_path / 'primg_gbsa_input-out.maegz').write_text('old output')
    completed = tmp_path / 'queue-1.completed.json'
    completed.write_text('{}')
    primg_gbsa_queue.prepare_new_calculation(SimpleNamespace(primg_gbsa_receptor=None, schrodinger='/suite'), tmp_path)
    assert not (tmp_path / 'receptor.maegz').exists()
    assert not (tmp_path / 'primg_gbsa_input.maegz').exists()
    assert not (tmp_path / 'primg_gbsa_input-out.maegz').exists()
    assert completed.exists()


def test_no_ligprep_ids_does_not_turn_all_compounds_into_terminal_failures(tmp_path, monkeypatch):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('REF','C',0,'reference')")
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,similar_to) VALUES ('NEW','C',1,'ligprepped',1)")
    prepared = tmp_path / 'iter1/ligprep/ligprep.maegz'
    prepared.parent.mkdir(parents=True)
    prepared.write_text('obsolete IDs')
    args = SimpleNamespace(db_path=db, output=tmp_path, schrodinger=Path('/suite'))
    monkeypatch.setattr(_core, 'assign_reference_compounds', lambda *a: None)
    monkeypatch.setattr(glide, 'active_jobs', lambda *a: [])
    monkeypatch.setattr(_core, 'split_ligprep_by_reference', lambda *a: {})
    with pytest.raises(RuntimeError, match='no current compound IDs'):
        glide._run_stage(args, SimpleNamespace(iteration=1, poll_interval=1, completion_fraction=.95, tail_timeout=1))
    with database.connect(db) as conn:
        assert conn.execute('SELECT state,failed_stage FROM compound WHERE iteration=1').fetchone() == ('glide_running', None)


@pytest.mark.parametrize('case', ['unsubmitted', 'done', 'running', 'offline'])
def test_obsolete_queue_reconciles_interrupted_child_without_submitting(tmp_path, monkeypatch, case):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,docking_score) VALUES ('current','C',2,'docked',-9)")
    root = tmp_path / 'primg_gbsa'
    root.mkdir()
    record = {'mode': primg_gbsa_queue.MODE, 'compound_ids': [30025],
              'run_dir': 'queue-1', 'entries': {'30025': 'active'}}
    primg_gbsa_async._save(root, record)
    child = root / 'queue-1/CMPID_30025'
    child.mkdir(parents=True)
    saved = {'command': ['/suite/prime_mmgbsa', str(child / 'primg_gbsa_input.maegz')],
             'compound_ids': [30025], 'job_id': None, 'launching': True,
             'attempt': 1, 'retry_at': 0, 'log_snapshot': {},
             'launch_started_at': time.time() - 120, 'launch_host': socket.gethostname()}
    primg_gbsa_async._save(child, saved)
    monkeypatch.setattr(primg_gbsa_async, '_launch_process_alive', lambda *a: False)
    def history(*a):
        if case == 'offline':
            raise OSError('JobServer unavailable')
        return [] if case == 'unsubmitted' else [{'jobId': 'recovered', 'status': case.upper()}]
    monkeypatch.setattr(primg_gbsa_async, '_launch_history', history)
    monkeypatch.setattr(primg_gbsa_async, '_job_details', lambda *a: [
        {'jobId': 'recovered', 'status': case.upper()}])
    monkeypatch.setattr(primg_gbsa_async, '_run_attempt', lambda *a: pytest.fail('obsolete compound submitted'))
    if case in ('unsubmitted', 'done'):
        assert primg_gbsa_async.reconcile_saved_job(root, db, 2) is None
        assert not primg_gbsa_async.job_file(root).exists()
        archived = next(tmp_path.glob('primg_gbsa.stale-*')) / 'queue-1/CMPID_30025'
        recovered = primg_gbsa_async.load_job(archived)
        assert recovered['launching'] is False
        assert recovered['job_id'] == (None if case == 'unsubmitted' else 'recovered')
    else:
        with pytest.raises(primg_gbsa_async.PrimeReconciliationPending):
            primg_gbsa_async.reconcile_saved_job(root, db, 2)
        assert primg_gbsa_async.load_job(root) == record
        assert not list(tmp_path.glob('primg_gbsa.stale-*'))
    with database.connect(db) as conn:
        assert conn.execute('SELECT state,docking_score,prime_gbsa_score FROM compound').fetchone() == ('docked', -9, None)
