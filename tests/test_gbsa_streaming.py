import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, gbsa_streaming, prime_async
from molnova.stages import mmgbsa
from molnova.stages._logging import timed_stage, work_started, format_elapsed, job_elapsed


def project(tmp_path):
    db=tmp_path/'project.sqlite'
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany("INSERT INTO compound(name,smiles,iteration,docking_score,state) "
                         "VALUES (?,'C',1,-9,'gbsa_running')", [('first',),('second',)])
    directory=tmp_path/'iter1/mmgbsa';directory.mkdir(parents=True)
    record={'command':['/suite/prime_mmgbsa','mmgbsa_input.maegz'], 'compound_ids':[1,2],
            'job_id':'parent-job','attempt':1,'launching':False,'retry_at':0}
    prime_async._save(directory,record)
    return SimpleNamespace(db_path=db,output=tmp_path,schrodinger=Path('/suite'),
                           target_count=2,gbsa_input_count=2,gbsa_elite_count=1),directory,record


def jobs(status='RUNNING'):
    return [{'jobId':'parent-job','jobName':'mmgbsa_input','status':status},
            {'jobId':'child-1','parentJobId':'parent-job','jobName':'mmgbsa_input.00001','status':'DONE',
             'timeStarted':'2026-10-10T01:00:00Z','statusUpdated':'2026-10-10T01:00:30Z'},
            {'jobId':'child-2','parentJobId':'parent-job','jobName':'mmgbsa_input.00002','status':'RUNNING'}]


def mock_results(monkeypatch, ids=(1,), unavailable=False):
    calls=[]
    def run(command, **kw):
        calls.append(command)
        if command[1]=='info':
            return SimpleNamespace(stdout='\n'.join(json.dumps(job) for job in jobs()))
        assert command[1]=='tail-file'
        if not unavailable:kw['stdout'].write(b'immutable completed child result')
        return SimpleNamespace(returncode=1 if unavailable else 0,stderr=b'')
    def extract(command,cwd):
        Path(command[-1]).write_text('id\tgbsa_score\n'+''.join(f'{cid}\t-40\n' for cid in ids))
    monkeypatch.setattr(gbsa_streaming.subprocess,'run',run)
    monkeypatch.setattr(_core,'run_command',extract)
    return calls


def test_running_parent_imports_completed_child_once_and_survives_restart(tmp_path,monkeypatch,capsys):
    args,directory,record=project(tmp_path)
    calls=mock_results(monkeypatch)
    monkeypatch.setattr(prime_async,'_run_attempt',lambda *a:pytest.fail('duplicate submission'))
    mmgbsa._run_stage(args,1)
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute('SELECT state,gbsa_score FROM compound ORDER BY id').fetchall()==[
            ('gbsa_done',-40),('gbsa_running',None)]
    assert database.iteration_completion_reason(args.db_path,1,2,2,1) is not None
    assert prime_async.load_job(directory)['imported_subjobs']==['child-1']
    mmgbsa._run_stage(args,1)
    assert sum(command[1]=='tail-file' for command in calls)==1
    assert 'calculation elapsed=00:00:30.0' in capsys.readouterr().out
    assert prime_async.load_job(directory)['job_id']=='parent-job'


def test_output_not_yet_available_retains_claim_and_retries_next_poll(tmp_path,monkeypatch):
    args,directory,record=project(tmp_path)
    mock_results(monkeypatch,unavailable=True)
    assert gbsa_streaming.collect_completed_subjobs(args,1,directory,record,jobs())==0
    assert not record.get('imported_subjobs')
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute('SELECT gbsa_score FROM compound').fetchall()==[(None,),(None,)]
    mock_results(monkeypatch)
    assert gbsa_streaming.collect_completed_subjobs(args,1,directory,record,jobs())==1


def test_only_matching_done_children_are_read(tmp_path,monkeypatch):
    args,directory,record=project(tmp_path)
    calls=mock_results(monkeypatch)
    snapshot=jobs()+[
        {'jobId':'foreign','parentJobId':'other-job','jobName':'mmgbsa_input.00003','status':'DONE'},
        {'jobId':'receptor','parentJobId':'parent-job','jobName':'receptor','status':'DONE'}]
    gbsa_streaming.collect_completed_subjobs(args,1,directory,record,snapshot)
    assert len(calls)==1
    assert calls[0][-1]=='parent-job'


def test_unsubmitted_ids_rejected_before_db_write_or_checkpoint(tmp_path,monkeypatch):
    args,directory,record=project(tmp_path)
    mock_results(monkeypatch,ids=(1,99))
    with pytest.raises(ValueError,match='unsubmitted'):
        gbsa_streaming.collect_completed_subjobs(args,1,directory,record,jobs())
    assert not record.get('imported_subjobs')
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute('SELECT gbsa_score FROM compound').fetchall()==[(None,),(None,)]


def test_db_commit_before_checkpoint_failure_is_idempotent(tmp_path,monkeypatch):
    args,directory,record=project(tmp_path)
    mock_results(monkeypatch)
    original=gbsa_streaming._save
    monkeypatch.setattr(gbsa_streaming,'_save',lambda *a:(_ for _ in ()).throw(OSError('interrupted checkpoint')))
    with pytest.raises(OSError):
        gbsa_streaming.collect_completed_subjobs(args,1,directory,record,jobs())
    monkeypatch.setattr(gbsa_streaming,'_save',original)
    recovered=prime_async.load_job(directory)
    assert gbsa_streaming.collect_completed_subjobs(args,1,directory,recovered,jobs())==0
    assert recovered['imported_subjobs']==['child-1']
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute('SELECT state,gbsa_score FROM compound WHERE id=1').fetchone()==('gbsa_done',-40)


@pytest.mark.parametrize('error',[RuntimeError('batch failed'),prime_async.PrimeAtomTypingError(2)])
def test_later_batch_failure_preserves_streamed_score_and_state(tmp_path,monkeypatch,error):
    args,directory,record=project(tmp_path)
    def fail(**kw):
        scores=tmp_path/'scores.tsv';scores.write_text('id\tgbsa_score\n1\t-40\n')
        _core.update_gbsa_scores(scores,args.db_path,allowed_ids=[1,2])
        prime_async.job_file(directory).unlink()
        raise error
    monkeypatch.setattr(_core,'run_iteration_mmgbsa',fail)
    if isinstance(error,prime_async.PrimeAtomTypingError):mmgbsa._run_stage(args,1)
    else:
        with pytest.raises(RuntimeError):mmgbsa._run_stage(args,1)
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute('SELECT state,gbsa_score FROM compound WHERE id=1').fetchone()==('gbsa_done',-40)


def test_parent_completion_uses_final_output_without_child_transfers(tmp_path,monkeypatch):
    args,directory,record=project(tmp_path)
    monkeypatch.setattr(gbsa_streaming.subprocess,'run',lambda *a,**kw:pytest.fail('obsolete child transfer'))
    assert gbsa_streaming.collect_completed_subjobs(args,1,directory,record,jobs('DONE')) is None


@pytest.mark.parametrize('failed',[False,True])
def test_stage_timer_reports_work_and_failure_but_not_idle(monkeypatch,capsys,failed):
    clock=iter([10,75.25]);monkeypatch.setattr('molnova.stages._logging.time.monotonic',lambda:next(clock))
    @timed_stage('Generate stage')
    def stage(active):
        if not active:return
        work_started('starting')
        if failed:raise RuntimeError('failed')
    stage(False);assert capsys.readouterr().out==''
    if failed:
        with pytest.raises(RuntimeError):stage(True)
    else:stage(True)
    output=capsys.readouterr().out
    assert 'elapsed=00:01:05.2' in output
    assert ('failed/interrupted' if failed else 'completed') in output


def test_external_job_duration_uses_server_timestamps():
    assert job_elapsed(jobs()[1])==30
    assert job_elapsed({}) is None
    assert format_elapsed(3661.5)=='01:01:01.5'
