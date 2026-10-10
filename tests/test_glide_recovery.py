import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from molnova import glide_recovery as recovery, schrodinger_guard as guard
from molnova.stages import glide
from molnova import prime_async as prime


def files(tmp_path):
    log=tmp_path/'glide_constrained.log';out=tmp_path/'glide_constrained_lib.maegz'
    log.write_text('JobId: old\nExitStatus: failed\nFinished at: previous job\n')
    out.write_text('previous poses')
    return log,out


def test_new_live_job_overrides_old_finished_log_and_output(tmp_path,monkeypatch):
    log,out=files(tmp_path)
    live=[{'jobId':'new','jobName':'glide_constrained','status':'RUNNING',
           'commandLine':f'/suite/glide {tmp_path}/glide_constrained.in','timeCreated':'new'}]
    assert recovery.group_status('/suite',tmp_path,log,out,live)=='RUNNING'
    assert recovery.load_record(tmp_path)['job_id']=='new'
    assert out.read_text()=='previous poses'


def test_parent_or_child_running_blocks_completion_and_download(tmp_path,monkeypatch):
    log,out=files(tmp_path)
    recovery.save_record(tmp_path,{'job_id':'new','downloaded':False})
    for parent_status,child_status in [('RUNNING','DONE'),('DONE','RUNNING')]:
        monkeypatch.setattr(recovery,'job_details',lambda *a:[
            {'jobId':'new','status':parent_status},{'jobId':'child','status':child_status}])
        monkeypatch.setattr(recovery.subprocess,'run',lambda *a,**kw:pytest.fail('Downloaded live output'))
        assert recovery.group_status('/suite',tmp_path,log,out,[])=='RUNNING'


def test_finished_saved_job_downloads_its_output_once_before_import(tmp_path,monkeypatch):
    log,out=files(tmp_path)
    recovery.save_record(tmp_path,{'job_id':'new','downloaded':False})
    monkeypatch.setattr(recovery,'job_details',lambda *a:[{'jobId':'new','status':'DONE'}])
    calls=[]
    def download(command,**kw):
        calls.append(command);out.write_text('new poses');log.write_text('JobId: new\nFinished at: now')
    monkeypatch.setattr(recovery.subprocess,'run',download)
    assert recovery.group_status('/suite',tmp_path,log,out,[])=='DONE'
    assert out.read_text()=='new poses'
    assert recovery.group_status('/suite',tmp_path,log,out,[])=='DONE'
    assert len(calls)==1
    assert calls[0][-1]=='new'


def test_unknown_launch_cannot_use_old_output_or_resubmit(tmp_path,monkeypatch):
    log,out=files(tmp_path)
    recovery.save_record(tmp_path,{'job_id':None})
    monkeypatch.setattr(guard,'active_jobs',lambda *a:[])
    monkeypatch.setattr(recovery,'active_jobs',lambda *a:[])
    assert recovery.group_status('/suite',tmp_path,log,out,[])=='UNKNOWN'
    with pytest.raises(RuntimeError,match='no saved JobId'):
        recovery.submit('/suite',['/suite/glide','input.in'],tmp_path)
    assert out.read_text()=='previous poses'


def test_launch_record_exists_before_native_submit_and_retains_identity(tmp_path,monkeypatch):
    log,out=files(tmp_path)
    monkeypatch.setattr(guard,'active_jobs',lambda *a:[])
    monkeypatch.setattr(recovery,'active_jobs',lambda *a:[])
    def launch(command,**kw):
        assert recovery.load_record(tmp_path)['job_id'] is None
        assert not out.exists() and not log.exists()
        return SimpleNamespace(stdout='JobId: new\n',stderr='')
    monkeypatch.setattr(recovery.subprocess,'run',launch)
    assert recovery.submit('/suite',['/suite/glide','input.in'],tmp_path)=='new'
    assert recovery.load_record(tmp_path)['job_id']=='new'
    assert len(list(tmp_path.glob('previous-*/glide_constrained_lib.maegz')))==1


def test_opposite_stage_fences_glide_before_files_are_archived(tmp_path,monkeypatch):
    log,out=files(tmp_path)
    monkeypatch.setattr(guard,'active_jobs',lambda *a:[
        {'jobId':'prime','jobName':'mmgbsa_input','status':'RUNNING'}])
    with pytest.raises(guard.SubmissionBlocked):
        recovery.submit('/suite',['/suite/glide','input.in'],tmp_path)
    assert recovery.load_record(tmp_path) is None
    assert out.read_text()=='previous poses'


def test_changed_input_never_deletes_live_or_uncertain_job_artifacts(tmp_path):
    prepared=tmp_path/'input.maegz';prepared.write_text('old')
    root=tmp_path/'glide';root.mkdir();glide.prepare_glide_files(root,prepared)
    group=root/'ref_1';group.mkdir();log,out=files(group)
    prepared.write_text('new')
    live=[{'spec':{'launchParams':{'launchDirectory':str(group)}}}]
    with pytest.raises(RuntimeError,match='live Glide'):
        glide.prepare_glide_files(root,prepared,live)
    recovery.save_record(group,{'job_id':None})
    with pytest.raises(RuntimeError,match='uncertain'):
        glide.prepare_glide_files(root,prepared)
    assert out.exists() and log.exists()


@pytest.mark.parametrize('timeout', [0, 1800])
def test_stage_does_not_import_partial_output_or_mark_failed_from_child_log(tmp_path,monkeypatch,timeout):
    import sqlite3
    from molnova import _core as c
    db=tmp_path/'results.sqlite'
    with sqlite3.connect(db) as conn:
        c.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('REF','C',0,'reference')")
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,similar_to) VALUES ('NEW','C',1,'glide_running',1)")
    args=SimpleNamespace(db_path=db,output=tmp_path,schrodinger=Path('/suite'),glide_job_timeout=timeout)
    prepared=tmp_path/'iter1/ligprep/ligprep_all.maegz';prepared.parent.mkdir(parents=True);prepared.write_text('input')
    group=tmp_path/'iter1/glide/ref_1';group.mkdir(parents=True)
    glide.prepare_glide_files(group.parent,prepared)
    log,out=files(group);ligand=group/'ligprep_group.maegz';ligand.write_text('ligand')
    live=[{'jobId':'new','jobName':'glide_constrained','status':'RUNNING','commandLine':f'/suite/glide {group}/glide_constrained.in'}]
    monkeypatch.setattr(glide,'active_jobs',lambda *a:live)
    stopped = []
    monkeypatch.setattr(recovery, 'job_details', lambda *a: [
        {'jobId': 'new', 'status': 'STOPPED' if stopped else 'RUNNING',
         'timeStarted': '1970-01-01T00:00:10Z'}])
    def native_stop(command, **kw):
        assert command == ['/suite/jsc', 'stop', '--force', 'new']
        stopped.append(True)
    monkeypatch.setattr(recovery.subprocess, 'run', native_stop)
    monkeypatch.setattr(c,'assign_reference_compounds',lambda *a:None)
    monkeypatch.setattr(c,'split_ligprep_by_reference',lambda *a:{1:ligand})
    monkeypatch.setattr(c,'extract_scores_and_best_poses',lambda *a:pytest.fail('Read old output'))
    monkeypatch.setattr(c,'submit_glide',lambda *a,**k:pytest.fail('Duplicate submission'))
    monkeypatch.setattr(glide.time,'sleep',lambda *a:None)
    import itertools
    monkeypatch.setattr(glide.time,'monotonic',itertools.count().__next__)
    ns=SimpleNamespace(iteration=1,poll_interval=1,completion_fraction=0,tail_timeout=0)
    glide._run_stage(args,ns)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT state,docking_score FROM compound WHERE iteration=1').fetchone()==('failed' if timeout else 'glide_running',None)
        if timeout:
            assert '1800s' in conn.execute('SELECT failure_message FROM compound WHERE iteration=1').fetchone()[0]


def test_legacy_prime_retry_fenced_before_launch_marker(tmp_path,monkeypatch):
    monkeypatch.setattr(guard,'active_jobs',lambda *a:[{'jobId':'g','jobName':'glide_constrained','status':'RUNNING'}])
    monkeypatch.setattr(prime,'_run_attempt',lambda *a:pytest.fail('Prime submitted while Glide active'))
    cmd=['/suite/prime_mmgbsa','input.maegz']
    assert prime.submit_or_poll(cmd,tmp_path,[1],submission_fence=lambda:guard.submission_guard('/suite','mmgbsa')) is False
    assert prime.load_job(tmp_path) is None  # No launch marker for a blocked submission.


@pytest.mark.parametrize('parent_status', ['RUNNING', 'DONE'])
def test_timeout_stops_only_live_tree_and_waits_for_children(tmp_path, monkeypatch, parent_status):
    log, out = files(tmp_path)
    recovery.save_record(tmp_path, {'job_id': 'parent', 'submitted_at': 1})
    jobs = [{'jobId': 'parent', 'status': parent_status, 'timeStarted': '1970-01-01T00:00:10Z'},
            {'jobId': 'child', 'status': 'RUNNING'}, {'jobId': 'finished', 'status': 'DONE'}]
    monkeypatch.setattr(recovery, 'job_details', lambda *a: jobs)
    monkeypatch.setattr(recovery.time, 'time', lambda: 1810)
    calls = []
    def stop(command, **kw):
        assert recovery.load_record(tmp_path)['timeout_requested_at'] == 1810
        calls.append(command)
    monkeypatch.setattr(recovery.subprocess, 'run', stop)
    assert recovery.group_status('/suite', tmp_path, log, out, [], timeout_seconds=1800) == 'RUNNING'
    assert calls == [['/suite/jsc', 'stop', '--force', *(['parent'] if parent_status == 'RUNNING' else []), 'child']]
    # Worker restart with the timeout disabled still reconciles the saved stop.
    assert recovery.group_status('/suite', tmp_path, log, out, [], timeout_seconds=0) == 'RUNNING'
    assert len(calls) == 1
    jobs[0]['status'] = 'STOPPED'
    jobs[1]['status'] = 'STOPPED'
    assert recovery.group_status('/suite', tmp_path, log, out, [], timeout_seconds=0) == 'STOPPED'
    assert len(calls) == 1  # Stale/partial output was never downloaded or imported.
    assert '1800s' in glide.failure_reason(tmp_path, 'STOPPED')


def test_timeout_uses_native_start_across_restart_and_does_not_stop_early(tmp_path, monkeypatch):
    log, out = files(tmp_path)
    recovery.save_record(tmp_path, {'job_id': 'parent', 'submitted_at': 1})
    monkeypatch.setattr(recovery.time, 'time', lambda: 1809)
    monkeypatch.setattr(recovery, 'job_details', lambda *a: [
        {'jobId': 'parent', 'status': 'RUNNING', 'timeStarted': '1970-01-01T00:00:10Z'}])
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **kw: pytest.fail('Stopped early'))
    assert recovery.group_status('/suite', tmp_path, log, out, [], timeout_seconds=1800) == 'RUNNING'
    assert 'timeout_requested_at' not in recovery.load_record(tmp_path)


def test_timeout_stop_failure_keeps_durable_intent_and_claims(tmp_path, monkeypatch):
    log, out = files(tmp_path)
    recovery.save_record(tmp_path, {'job_id': 'parent', 'submitted_at': 1})
    monkeypatch.setattr(recovery.time, 'time', lambda: 1801)
    monkeypatch.setattr(recovery, 'job_details', lambda *a: [{'jobId': 'parent', 'status': 'RUNNING'}])
    def fail(*a, **kw):
        raise OSError('JobServer unavailable')
    monkeypatch.setattr(recovery.subprocess, 'run', fail)
    with pytest.raises(OSError):
        recovery.group_status('/suite', tmp_path, log, out, [], timeout_seconds=1800)
    assert recovery.load_record(tmp_path)['timeout_requested_at'] == 1801
    assert 'timeout_stop_sent_at' not in recovery.load_record(tmp_path)
