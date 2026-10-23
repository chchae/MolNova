"""Restart recovery tests; native job mocks do not validate MD or GBSA science."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, driver, md_gbsa_retry as remote, score_recovery
from molnova.paths import stage_directory
from molnova.states import CompoundState as State


@pytest.fixture
def project(tmp_path):
    db = tmp_path/'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        conn.executemany('INSERT INTO compound(name,smiles,iteration,state,docking_score,prime_gbsa_score,failed_stage) '
                         'VALUES (?,\'C\',?,?,?,?,?)',[
            ('old',0,State.FAILED,-12,None,'prime_gbsa'),
            ('good',1,State.PRIME_GBSA_DONE,-11,-50,None),
            ('retry',1,State.FAILED,-10,None,'prime_gbsa'),
            ('outside',1,State.FAILED,-1,None,'prime_gbsa'),
            ('upstream',1,State.FAILED,-9,None,'ligprep')])
    return SimpleNamespace(db_path=db,output=tmp_path,prime_gbsa_enabled=True,
                           prime_gbsa_input_count=2,target_count=4,max_iteration=3)


def test_only_latest_missing_selected_prime_is_requested_once(project):
    assert database.request_latest_score_retries(project)==1
    requests=database.score_retry_requests(project.db_path,1,'prime_gbsa')
    assert len(requests)==1 and requests[0][0]==3
    assert not database.score_retry_requests(project.db_path,0,'prime_gbsa')
    database.request_latest_score_retries(project)
    assert requests==database.score_retry_requests(project.db_path,1,'prime_gbsa')
    assert driver.choose_iteration(project)==1
    assert 'recovery' in database.stage_completion_reason(project.db_path,1,'prime_gbsa',4,2)


def test_retry_rearms_one_compound_and_not_again_during_polling(project):
    database.request_latest_score_retries(project)
    cid,token=database.score_retry_requests(project.db_path,1,'prime_gbsa')[0]
    database.finish_score_retry(project.db_path,1,'prime_gbsa',cid,token)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT state,prime_gbsa_score FROM compound WHERE id=2').fetchone()==(State.PRIME_GBSA_DONE,-50)
        assert conn.execute('SELECT state,failed_stage FROM compound WHERE id=3').fetchone()==(State.DOCKED,None)
        conn.execute("UPDATE compound SET state='failed',failed_stage='prime_gbsa' WHERE id=3")
    assert not database.score_retry_requests(project.db_path,1,'prime_gbsa')
    database.request_latest_score_retries(project)
    assert database.score_retry_requests(project.db_path,1,'prime_gbsa')[0][1]!=token


def test_disabled_prime_and_explicit_other_iteration_are_not_retried(project):
    assert database.request_latest_score_retries(project,0) is None
    project.prime_gbsa_enabled=False
    assert database.request_latest_score_retries(project) is None


def test_reference_iteration_zero_can_be_retried(project):
    with database.connect(project.db_path) as conn:
        conn.execute('DELETE FROM compound WHERE iteration=1')
    assert database.request_latest_score_retries(project)==0
    requests=database.score_retry_requests(project.db_path,0,'prime_gbsa')
    assert [r[0] for r in requests]==[1]
    assert driver.choose_iteration(project)==0
    assert 'recovery' in database.reference_prime_gbsa_completion_reason(project.db_path)
    database.finish_score_retry(project.db_path,0,'prime_gbsa',*requests[0])
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT state FROM compound WHERE id=1').fetchone()[0]==State.REFERENCE


def test_late_score_commit_is_preserved(project):
    database.request_latest_score_retries(project)
    cid,token=database.score_retry_requests(project.db_path,1,'prime_gbsa')[0]
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state='prime_gbsa_done',prime_gbsa_score=-80 WHERE id=3")
    database.finish_score_retry(project.db_path,1,'prime_gbsa',cid,token)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT state,prime_gbsa_score FROM compound WHERE id=3').fetchone()==(State.PRIME_GBSA_DONE,-80)


def make_replica(tmp_path, md='FAILED', prime=None):
    for phase,status in [('md',md),('prime',prime)]:
        if status is None:
            continue
        folder=tmp_path/phase
        folder.mkdir()
        (folder/(phase+'.job.json')).write_text(json.dumps({'job_id':phase}))
        if phase=='prime':
            (folder/'frames.maegz').write_bytes(b'prepared frames')
    return {'md':md,'prime':prime}


@pytest.mark.parametrize('phase,md,prime,redo',[('md','FAILED',None,'md'),
                         ('prime','DONE','FAILED','prime'),('analysis','DONE',None,'md'),
                         ('validation','DONE','DONE','prime')])
def test_failed_phase_archived_and_successful_md_reused(tmp_path,monkeypatch,phase,md,prime,redo):
    statuses=make_replica(tmp_path,md,prime)
    monkeypatch.setattr(remote,'native_rows',lambda suite,jid:[{'jobId':jid,'status':statuses[jid]}])
    (tmp_path/'frames.json').write_text('[]')
    answer=remote.retry_replica(tmp_path,'/suite','retry1',phase)
    history=Path(answer['history'])
    assert (history/redo/(redo+'.job.json')).exists()
    assert not (tmp_path/redo/(redo+'.job.json')).exists()
    if redo=='prime':
        assert (tmp_path/'md/md.job.json').exists()
        assert (tmp_path/'prime/frames.maegz').read_bytes()==b'prepared frames'
        assert (tmp_path/'frames.json').exists()
    else:
        assert not (tmp_path/'frames.json').exists()
    assert remote.retry_replica(tmp_path,'/suite','retry1',phase)==answer


@pytest.mark.parametrize('bad_parent,bad_child',[('RUNNING',None),('FAILED','RUNNING'),('DONE','UNKNOWN')])
def test_active_or_uncertain_native_parent_and_children_never_reset(tmp_path,monkeypatch,bad_parent,bad_child):
    make_replica(tmp_path)
    rows=[{'jobId':'md','status':bad_parent}]
    if bad_child:
        rows.append({'jobId':'child','status':bad_child})
    monkeypatch.setattr(remote,'native_rows',lambda *args:rows)
    with pytest.raises(RuntimeError,match='active or uncertain'):
        remote.retry_replica(tmp_path,'/suite','retry1','md')
    assert (tmp_path/'md/md.job.json').exists()
    assert not (tmp_path/'retry_history').exists()


def test_uncertain_launch_and_missing_parent_never_reset(tmp_path,monkeypatch):
    make_replica(tmp_path)
    monkeypatch.setattr(remote,'native_rows',lambda *args:[{'jobId':'different','status':'DONE'}])
    with pytest.raises(RuntimeError):
        remote.retry_replica(tmp_path,'/suite','retry1','md')
    (tmp_path/'md/md.job.json').write_text('{"launching":true}')
    with pytest.raises(RuntimeError,match='no JobId'):
        remote.retry_replica(tmp_path,'/suite','retry1','md')


def test_complete_cached_native_result_is_recovered_without_new_job(tmp_path,monkeypatch):
    make_replica(tmp_path,'DONE','DONE')
    (tmp_path/'result.json').write_text('{"status":"done"}')
    monkeypatch.setattr(remote,'native_rows',lambda suite,jid:[{'jobId':jid,'status':'DONE'}])
    assert remote.retry_replica(tmp_path,'/suite','retry1','prime')=={'status':'recover'}
    assert (tmp_path/'md/md.job.json').exists()
    assert (tmp_path/'prime/prime.job.json').exists()


def test_crash_between_archive_moves_resumes_same_intent(tmp_path,monkeypatch):
    make_replica(tmp_path,'FAILED','FAILED')
    monkeypatch.setattr(remote,'native_rows',lambda suite,jid:[{'jobId':jid,'status':'FAILED'}])
    rename=remote.os.rename
    def interrupted(source,target):
        if Path(source).name=='prime':
            raise OSError('simulated interruption')
        return rename(source,target)
    monkeypatch.setattr(remote.os,'rename',interrupted)
    with pytest.raises(OSError):
        remote.retry_replica(tmp_path,'/suite','retry1','md')
    monkeypatch.setattr(remote.os,'rename',rename)
    remote.retry_replica(tmp_path,'/suite','retry1','md')
    assert (tmp_path/'retry_history/retry1/md/md.job.json').exists()
    assert (tmp_path/'retry_history/retry1/prime/prime.job.json').exists()


def test_prime_retry_requires_verified_terminal_evidence(project,monkeypatch):
    from molnova import prime_gbsa_async as prime
    database.request_latest_score_retries(project)
    folder=stage_directory(project.output,1,'prime_gbsa')
    folder.mkdir(parents=True)
    record={'compound_ids':[3],'job_id':'native','command':['/suite/prime_mmgbsa']}
    (folder/'batch-1.failed.json').write_text(json.dumps(record))
    monkeypatch.setattr(prime,'_job_details',lambda *a:[{'jobId':'native','status':'RUNNING'}])
    with pytest.raises(RuntimeError,match='active or uncertain'):
        score_recovery.recover_prime(project,1,folder)
    assert database.score_retry_requests(project.db_path,1,'prime_gbsa')
    monkeypatch.setattr(prime,'_job_details',lambda *a:[{'jobId':'native','status':'FAILED'}])
    score_recovery.recover_prime(project,1,folder)
    assert not database.score_retry_requests(project.db_path,1,'prime_gbsa')
    assert list((folder/'retry_history').glob('*/terminal-evidence.json'))


def test_md_retry_keeps_frozen_selection_and_successes(project,monkeypatch):
    with database.connect(project.db_path) as conn:
        database.ensure_md_gbsa_schema(conn)
        conn.execute("UPDATE compound SET prime_gbsa_score=-50,state='failed',failed_stage='md_gbsa' WHERE id=3")
        conn.execute('INSERT INTO iteration_md_gbsa_selection VALUES (1,?,?)',
                     (json.dumps([2,3]),json.dumps(dict(root='/run',suite='/suite',ssh_host='host',replicates=1))))
        conn.execute("UPDATE compound SET md_gbsa_score=-70,state='md_gbsa_done' WHERE id=2")
    database.request_latest_score_retries(project)
    assert [r[0] for r in database.score_retry_requests(project.db_path,1,'md_gbsa')]==[3]
    directory=stage_directory(project.output,1,'md_gbsa');directory.mkdir(parents=True)
    progress={'(2, 0)':{'status':'done'},'(3, 0)':{'status':'failed','phase':'prime'}}
    (directory/'progress.json').write_text(json.dumps(progress))
    from molnova import md_gbsa
    calls=[]
    monkeypatch.setattr(md_gbsa,'ssh',lambda host,command: calls.append(command) or '{"status":"ready"}')
    score_recovery.recover_md(project,1,database.md_gbsa_selection(project.db_path,1))
    assert len(calls)==1
    assert json.loads((directory/'progress.json').read_text())=={'(2, 0)':{'status':'done'}}
    assert database.md_gbsa_selection(project.db_path,1)['ids']==[2,3]
    assert database.md_gbsa_rows(project.db_path,1)[2][1]==-70
    assert database.md_gbsa_rows(project.db_path,1)[3][2]==State.MD_GBSA_RUNNING


@pytest.mark.parametrize('iteration',[0,1])
def test_any_md_score_suppresses_all_prime_backfill_in_that_cycle(project,iteration):
    with database.connect(project.db_path) as conn:
        conn.execute('DELETE FROM compound WHERE iteration!=?',(iteration,))
        if iteration==0:
            conn.execute("INSERT INTO compound(name,smiles,iteration,state,md_gbsa_score) VALUES ('md','C',0,'md_gbsa_done',-70)")
        else:
            conn.execute("UPDATE compound SET state='md_gbsa_done',md_gbsa_score=-70,prime_gbsa_score=NULL WHERE id=2")
    assert database.request_latest_score_retries(project) is None
    assert not database.score_retry_requests(project.db_path,iteration,'prime_gbsa')
    assert database.prime_gbsa_candidates(project.db_path,iteration,2,1)==[]
    if iteration==0:
        assert database.reference_prime_gbsa_completion_reason(project.db_path) is None
    else:
        assert database.iteration_completion_reason(project.db_path,1,4,2,0) is None


def test_existing_prime_retry_intent_is_closed_without_rearming_when_md_begins(project,monkeypatch):
    database.request_latest_score_retries(project)
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state='md_gbsa_done',md_gbsa_score=-70 WHERE id=2")
    directory=stage_directory(project.output,1,'prime_gbsa')
    score_recovery.recover_prime(project,1,directory)
    assert not database.score_retry_requests(project.db_path,1,'prime_gbsa')
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT state,failed_stage,prime_gbsa_score FROM compound WHERE id=3').fetchone()==(State.FAILED,'prime_gbsa',None)


def stages():
    return driver.build_stages(False,False,True,True)


def enter_md(project):
    with database.connect(project.db_path) as conn:
        conn.execute('INSERT INTO iteration_md_gbsa_selection VALUES (1,?,?)',(json.dumps([2]),'{}'))


def test_resume_last_md_stage_without_earlier_workers(project):
    enter_md(project)
    assert driver.resume_iteration_stages(project,1,stages())==[('md-gbsa','molnova.stages.md_gbsa')]
    assert driver.resume_iteration_stages(project,0,stages())==stages()
    assert driver.resume_iteration_stages(project,2,stages())==stages()


def test_resume_prime_when_md_has_not_started(project):
    assert [s for s,_ in driver.resume_iteration_stages(project,1,stages())]==['prime_gbsa','md-gbsa']


def test_resume_md_retains_saved_prime_recovery_without_new_prime_stage(project):
    enter_md(project)
    directory=stage_directory(project.output,1,'prime_gbsa');directory.mkdir(parents=True)
    (directory/'prime_gbsa_job.json').write_text('{"job_id":"existing"}')
    assert [s for s,_ in driver.resume_iteration_stages(project,1,stages())]==['prime_gbsa-recovery','md-gbsa']


def test_resume_md_retains_unknown_glide_and_upstream_claims(project):
    enter_md(project)
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state='amber_running' WHERE id=4")
    directory=stage_directory(project.output,1,'glide')/'group01';directory.mkdir(parents=True)
    (directory/'glide_job.json').write_text('{"job_id":"existing","downloaded":false}')
    assert [s for s,_ in driver.resume_iteration_stages(project,1,stages())]==['glide','amber-gbsa-recovery','md-gbsa']
    (directory/'glide_job.json').write_text('{"job_id":"existing","downloaded":true}')
    assert [s for s,_ in driver.resume_iteration_stages(project,1,stages())]==['amber-gbsa-recovery','md-gbsa']


def test_old_prime_intents_are_closed_once_md_selection_is_frozen(project):
    database.request_latest_score_retries(project)
    assert database.score_retry_requests(project.db_path,1,'prime_gbsa')
    enter_md(project)
    database.request_latest_score_retries(project)
    assert not database.score_retry_requests(project.db_path,1,'prime_gbsa')


def test_new_cycle_still_has_prime_candidates_after_previous_cycle_md(project):
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET md_gbsa_score=-70,state='md_gbsa_done' WHERE iteration=0")
        conn.execute("UPDATE compound SET state='docked',failed_stage=NULL WHERE id=3")
    assert database.prime_gbsa_candidates(project.db_path,1,2,4)==[3]
