"""Legacy Adam JobControl recovery; no scientific calculation in these tests."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from molnova import md_gbsa_jobcontrol as adapter, md_gbsa_native as native


def job(identity, *, children=(), complete=True, exit_status='finished', directory='/launch', parent=''):
    return SimpleNamespace(JobId=identity, ParentJobId=parent, SubJobs=children,
        isComplete=lambda:complete, Status='completed' if complete else 'exited',
        ExitStatus=exit_status, isDownloaded=lambda:complete, Name='prime_mmgbsa',
        Commandline='prime frames', Dir=directory, JobHost='adam', LaunchTime='2026-11-02-12:45:27')


def api(monkeypatch, jobs):
    def load(identity):
        if identity not in jobs:
            raise RuntimeError('Legacy job record unavailable')
        return jobs[identity]
    monkeypatch.setitem(sys.modules, 'schrodinger', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'schrodinger.job', SimpleNamespace(jobcontrol=SimpleNamespace(Job=load)))


def test_completed_parent_does_not_hide_live_legacy_child(monkeypatch):
    api(monkeypatch, {'adam-0-1234abcd':job('adam-0-1234abcd',children=['adam-0-1234abce']),
        'adam-0-1234abce':job('adam-0-1234abce',complete=False,parent='adam-0-1234abcd')})
    assert [row['status'] for row in adapter.legacy_rows('adam-0-1234abcd')] == ['DONE','RUNNING']


def test_missing_legacy_child_fails_closed(monkeypatch):
    api(monkeypatch, {'adam-0-1234abcd':job('adam-0-1234abcd',children=['adam-0-1234abce'])})
    with pytest.raises(RuntimeError,match='unavailable'):
        adapter.legacy_rows('adam-0-1234abcd')


@pytest.mark.parametrize('exit_status,expected',[('finished','DONE'),('died','FAILED')])
def test_zero_client_exit_is_not_legacy_success(monkeypatch,exit_status,expected):
    api(monkeypatch, {'adam-0-1234abcd':job('adam-0-1234abcd',exit_status=exit_status)})
    assert adapter.legacy_rows('adam-0-1234abcd')[0]['status'] == expected


@pytest.mark.parametrize('text,valid',[
    ('** no active jobs were found **',True), ('connection failed',False),
    ("** no jobs were found that match 'active' **",True)])
def test_legacy_empty_listing_must_be_explicit(monkeypatch,text,valid):
    monkeypatch.setattr(adapter.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=text))
    if valid:
        assert adapter.legacy_ids(Path('/suite'),'active') == set()
    else:
        with pytest.raises(RuntimeError,match='Unrecognized'):
            adapter.legacy_ids(Path('/suite'),'active')


@pytest.mark.parametrize('complete',[False,True])
def test_saved_legacy_job_recovers_without_launch_or_jsc_download(tmp_path,monkeypatch,complete):
    identity='adam-0-1234abcd'
    api(monkeypatch, {identity:job(identity,complete=complete,directory=str(tmp_path))})
    native.save(tmp_path/'prime.job.json',dict(command=['prime','frames'],job_id=identity,client_exit=0))
    def forbidden(*a,**kw):
        pytest.fail('legacy recovery invoked native JobServer or relaunched')
    fake=SimpleNamespace(native_rows=forbidden,active_jobs=forbidden,poll_job=forbidden,
                         save=native.save,TERMINAL=native.TERMINAL)
    adapter.install(fake)
    monkeypatch.setattr(adapter.subprocess,'Popen',forbidden)
    result=fake.poll_job(tmp_path,'prime',['prime','frames'],Path('/suite'))
    assert result['status']==('done' if complete else 'running')
    saved=json.loads((tmp_path/'prime.job.json').read_text())
    assert saved['job_id']==identity
    assert bool(saved.get('downloaded'))==complete


def test_legacy_command_identity_mismatch_preserves_journal(tmp_path,monkeypatch):
    identity='adam-0-1234abcd'
    api(monkeypatch, {identity:job(identity,directory='/another-job')})
    native.save(tmp_path/'prime.job.json',dict(command=['prime','frames'],job_id=identity))
    fake=SimpleNamespace(native_rows=None,active_jobs=None,poll_job=None,save=native.save,TERMINAL=native.TERMINAL)
    adapter.install(fake)
    with pytest.raises(RuntimeError,match='directory/command mismatch'):
        fake.poll_job(tmp_path,'prime',['prime','frames'],Path('/suite'))
    assert not json.loads((tmp_path/'prime.job.json').read_text()).get('downloaded')


def test_lost_legacy_submission_reply_recovers_exact_launch_without_resubmission(tmp_path,monkeypatch):
    identity='adam-0-1234abcd'
    item=job(identity,directory=str(tmp_path))
    api(monkeypatch,{identity:item})
    import time
    started=time.mktime(time.strptime(item.LaunchTime,'%Y-%m-%d-%H:%M:%S'))
    native.save(tmp_path/'prime.job.json',dict(command=['prime','frames'],job_id=None,
                                            launching=True,started_at=started))
    monkeypatch.setattr(adapter,'legacy_ids',lambda *a:{identity})
    def forbidden(*a,**kw):
        pytest.fail('duplicate launch or invalid jsc recovery')
    fake=SimpleNamespace(native_rows=forbidden,active_jobs=forbidden,poll_job=forbidden,
                         save=native.save,TERMINAL=native.TERMINAL)
    adapter.install(fake)
    monkeypatch.setattr(adapter.subprocess,'Popen',forbidden)
    assert fake.poll_job(tmp_path,'prime',['prime','frames'],Path('/suite'))['status']=='done'
    assert json.loads((tmp_path/'prime.job.json').read_text())['job_id']==identity


def test_native_uuid_still_uses_original_native_status_and_polling(tmp_path):
    identity='df96e9ce-d60c-11f1-a66c-3cecef0ebb4e'
    native.save(tmp_path/'prime.job.json',dict(command=['prime','frames'],job_id=identity))
    fake=SimpleNamespace(native_rows=lambda *a:['native rows'],active_jobs=lambda *a:[],
        poll_job=lambda *a:dict(status='running',backend='native'),save=native.save,TERMINAL=native.TERMINAL)
    adapter.install(fake)
    assert fake.native_rows(Path('/suite'),identity)==['native rows']
    assert fake.poll_job(tmp_path,'prime',['prime','frames'],Path('/suite'))['backend']=='native'


def test_operational_adapter_preserves_frozen_helper_bytes(tmp_path):
    from molnova import md_gbsa_local as local
    settings=dict(root=str(tmp_path),suite='/suite',ssh_host='adam',helper=str(tmp_path/'md_gbsa_local_native.py'))
    native.save(tmp_path/'owner.json',dict(local=settings))
    (tmp_path/'md_gbsa_local_native.py').write_bytes(b'frozen scientific command')
    path=local.control_helper(settings)
    assert path.parent==tmp_path/'control'
    assert (tmp_path/'md_gbsa_local_native.py').read_bytes()==b'frozen scientific command'
    assert local.control_helper(settings)==path
