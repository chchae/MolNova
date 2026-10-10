import json
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest
from molnova import prime_async as prime, schrodinger_guard as guard
from molnova.schrodinger_retry import _run_attempt


def record(tmp_path):
    saved = {'command':['/suite/prime_mmgbsa', str(tmp_path/'mmgbsa_input.maegz')],
             'compound_ids':[15478], 'job_id':None, 'launching':True,
             'attempt':1, 'retry_at':0, 'log_snapshot':{},
             'launch_started_at':time.time()-120, 'launch_host':socket.gethostname()}
    prime._save(tmp_path, saved)
    return saved


def test_confirmed_unsubmitted_launch_released_without_immediate_unguarded_submit(tmp_path, monkeypatch):
    saved = record(tmp_path)
    monkeypatch.setattr(prime, '_launch_process_alive', lambda *a: False)
    monkeypatch.setattr(prime, '_launch_history', lambda *a: [])
    monkeypatch.setattr(prime, '_run_attempt', lambda *a: pytest.fail('unguarded launch'))
    assert prime.submit_or_poll([], tmp_path, [15478]) is False
    current = prime.load_job(tmp_path)
    assert current['launching'] is False and current['job_id'] is None
    assert current['attempt'] == 0
    assert json.loads(next(tmp_path.glob('prime_job.unsubmitted-*.json')).read_text()) == saved


@pytest.mark.parametrize('status', ['RUNNING','DONE','FAILED'])
def test_recover_native_job_even_if_slurm_queue_is_empty(tmp_path, monkeypatch, status):
    saved = record(tmp_path)
    monkeypatch.setattr(prime, '_launch_process_alive', lambda *a: False)
    monkeypatch.setattr(prime, '_launch_history', lambda *a: [
        {'jobId':'existing', 'status':status, 'jobName':'mmgbsa_input'}])
    assert prime.reconcile_interrupted_submission(tmp_path, saved)
    assert prime.load_job(tmp_path)['job_id']=='existing'
    assert not prime.load_job(tmp_path)['launching']


def test_recover_jobid_from_current_log(tmp_path, monkeypatch):
    saved = record(tmp_path)
    (tmp_path/'input.log').write_text('JobId: existing\n')
    monkeypatch.setattr(prime, '_launch_process_alive', lambda *a: False)
    monkeypatch.setattr(prime, '_launch_history', lambda *a: [])
    monkeypatch.setattr(guard, 'job_details', lambda *a: [{'jobId':'existing','status':'DONE'}])
    assert prime.reconcile_interrupted_submission(tmp_path,saved)
    assert saved['job_id']=='existing'


@pytest.mark.parametrize('cause', ['client','offline','artifact','grace','multiple','otherhost'])
def test_uncertain_launch_is_never_released(tmp_path, monkeypatch, cause):
    saved = record(tmp_path)
    monkeypatch.setattr(prime, '_launch_process_alive', lambda *a: cause=='client')
    def history(*a):
        if cause=='offline':raise OSError('offline')
        if cause=='multiple':return [{'jobId':'a'}, {'jobId':'b'}]
        return []
    monkeypatch.setattr(prime, '_launch_history', history)
    if cause=='artifact':(tmp_path/'input-out.maegz').write_text('unidentified result')
    if cause=='grace':saved['launch_started_at']=time.time()
    if cause=='otherhost':
        saved['launch_host']='another-host'
        monkeypatch.undo()
    prime._save(tmp_path,saved)
    with pytest.raises((prime.PrimeReconciliationPending,OSError)):
        prime.reconcile_interrupted_submission(tmp_path,saved)
    assert prime.load_job(tmp_path)['launching']
    assert not list(tmp_path.glob('prime_job.unsubmitted-*.json'))


def test_history_query_uses_terminal_jobs_and_empty_response_is_not_error(tmp_path, monkeypatch):
    saved=record(tmp_path); calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd,1,'','No jobs matching your search criteria were found.\n')
    monkeypatch.setattr(prime.subprocess,'run',run)
    assert prime._launch_history(tmp_path,saved)==[]
    assert '--any-status' in calls[0] and '--launch-dir' in calls[0]


def test_jobserver_error_cannot_be_treated_as_empty_history(tmp_path, monkeypatch):
    saved=record(tmp_path)
    monkeypatch.setattr(prime.subprocess,'run',lambda cmd,**kw:
        subprocess.CompletedProcess(cmd,1,'','JobServer connection refused'))
    with pytest.raises(subprocess.CalledProcessError):prime._launch_history(tmp_path,saved)


def test_jobid_is_persisted_before_submission_client_returns(tmp_path,monkeypatch):
    def launch(command,directory,on_line):
        on_line('JobId: durable-id\n')
        assert prime.load_job(tmp_path)['job_id']=='durable-id'
        raise OSError('client disconnected after JobId output')
    monkeypatch.setattr(prime,'_run_attempt',launch)
    with pytest.raises(OSError):
        prime.submit_or_poll(['/suite/prime_mmgbsa',str(tmp_path/'input.maegz')],tmp_path,[15478])
    assert prime.load_job(tmp_path)['job_id']=='durable-id'
    assert prime.load_job(tmp_path)['launching'] is False


def test_native_attempt_streams_jobid_to_checkpoint(tmp_path):
    import sys
    captured=[]
    result=_run_attempt([sys.executable,'-c',"print('JobId: example')"],tmp_path,captured.append)
    assert result==(0,'JobId: example\n')
    assert captured==['JobId: example\n']


def test_prime_waits_for_live_child_of_done_parent(tmp_path,monkeypatch):
    saved=record(tmp_path);saved.update(job_id='parent',launching=False)
    prime._save(tmp_path,saved)
    monkeypatch.setattr(prime,'_job_details',lambda *a:[
        {'jobId':'parent','status':'DONE'},{'jobId':'child','status':'RUNNING'}])
    monkeypatch.setattr(prime.subprocess,'run',lambda *a,**kw:pytest.fail('early download'))
    assert prime.submit_or_poll([],tmp_path,[15478]) is False
    assert prime.load_job(tmp_path)['job_id']=='parent'
