from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from molnova import schrodinger_guard as guard


@pytest.mark.parametrize('job', [
    {'jobName': 'primg_gbsa'},
    {'jobName': 'custom', 'commandLine': '/suite/prime_mmgbsa custom.maegz'},
    {'jobName': 'custom', 'spec': {'taskSpec': {'programName': 'Prime MMGBSA'}}},
])
def test_canonical_and_native_prime_jobs_block_glide(monkeypatch, job):
    assert guard.job_kind(job) == 'primg_gbsa'
    monkeypatch.setattr(guard, 'active_jobs', lambda *args: [dict(job, jobId='external')])
    with pytest.raises(guard.SubmissionBlocked, match='primg_gbsa still active'):
        with guard.submission_guard('/suite', 'glide'):
            pytest.fail('Submitted Glide while native Prime job is active')


def test_live_children_block_even_when_parent_has_finished(monkeypatch):
    def run(command, **kw):
        if command[1] == 'list':return SimpleNamespace(stdout='parent\n')
        return SimpleNamespace(stdout='\n'.join(json.dumps(j) for j in [
            {'jobId':'parent','status':'DONE','jobName':'glide'},
            {'jobId':'child','status':'RUNNING','jobName':'glide-0001'}]))
    monkeypatch.setattr(guard.subprocess,'run',run)
    with pytest.raises(guard.SubmissionBlocked,match='glide still active'):
        with guard.submission_guard(Path('/suite'),'primg_gbsa'):
            pytest.fail('primg_gbsa submitted while Glide child running')


@pytest.mark.parametrize('stage,other',[('primg_gbsa','glide'),('glide','primg_gbsa_input')])
def test_opposite_stage_blocked_across_projects(monkeypatch,stage,other):
    monkeypatch.setattr(guard,'active_jobs',lambda *a:[{'jobId':'external','jobName':other,'status':'RUNNING'}])
    with pytest.raises(guard.SubmissionBlocked):
        with guard.submission_guard('/suite',stage):pytest.fail('Overlap')


def test_unknown_jobserver_fails_closed(monkeypatch):
    def failure(*a):raise RuntimeError('offline')
    monkeypatch.setattr(guard,'active_jobs',failure)
    with pytest.raises(guard.SubmissionBlocked,match='Cannot verify'):
        with guard.submission_guard('/suite','glide'):pytest.fail('Unverified submission')


def test_submission_checks_are_atomic_and_release_after_error(monkeypatch):
    monkeypatch.setattr(guard,'active_jobs',lambda *a:[])
    with guard.submission_guard('/suite','glide'):
        with pytest.raises(guard.SubmissionBlocked,match='in progress'):
            with guard.submission_guard('/suite','primg_gbsa'):pytest.fail('Concurrent guard')
    with guard.submission_guard('/suite','primg_gbsa'):pass


def test_native_no_active_jobs_exit_one_is_an_empty_queue(monkeypatch):
    import subprocess
    monkeypatch.setattr(guard.subprocess, 'run', lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 1, '', 'No jobs matching your search criteria were found.\n'))
    assert guard.active_jobs('/suite') == []
    with guard.submission_guard('/suite','primg_gbsa'):pass


def test_native_lookup_error_is_not_mistaken_for_empty_queue(monkeypatch):
    import subprocess
    monkeypatch.setattr(guard.subprocess, 'run', lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 1, '', 'JobServer connection refused'))
    with pytest.raises(guard.SubmissionBlocked, match='Cannot verify'):
        with guard.submission_guard('/suite','primg_gbsa'):pytest.fail('Unverified empty queue')


def test_parent_details_avoid_duplicate_child_queries(monkeypatch):
    import subprocess
    calls = []
    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == 'list':
            return subprocess.CompletedProcess(cmd, 0, 'parent\nchild\n')
        return subprocess.CompletedProcess(cmd, 0, '\n'.join(json.dumps(job) for job in [
            {'jobId':'parent','status':'DONE'}, {'jobId':'child','status':'RUNNING'}]))
    monkeypatch.setattr(guard.subprocess, 'run', run)
    assert guard.active_jobs('/suite') == [{'jobId':'child','status':'RUNNING'}]
    assert len(calls) == 2
