from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from molnova import schrodinger_guard as guard


def test_live_children_block_even_when_parent_has_finished(monkeypatch):
    def run(command, **kw):
        if command[1] == 'list':return SimpleNamespace(stdout='parent\n')
        return SimpleNamespace(stdout='\n'.join(json.dumps(j) for j in [
            {'jobId':'parent','status':'DONE','jobName':'glide_constrained'},
            {'jobId':'child','status':'RUNNING','jobName':'glide_constrained-0001'}]))
    monkeypatch.setattr(guard.subprocess,'run',run)
    with pytest.raises(guard.SubmissionBlocked,match='glide still active'):
        with guard.submission_guard(Path('/suite'),'mmgbsa'):
            pytest.fail('MM-GBSA submitted while Glide child running')


@pytest.mark.parametrize('stage,other',[('mmgbsa','glide'),('glide','mmgbsa_input')])
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
            with guard.submission_guard('/suite','mmgbsa'):pytest.fail('Concurrent guard')
    with guard.submission_guard('/suite','mmgbsa'):pass


def test_native_no_active_jobs_exit_one_is_an_empty_queue(monkeypatch):
    import subprocess
    monkeypatch.setattr(guard.subprocess, 'run', lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 1, '', 'No jobs matching your search criteria were found.\n'))
    assert guard.active_jobs('/suite') == []
    with guard.submission_guard('/suite','mmgbsa'):pass


def test_native_lookup_error_is_not_mistaken_for_empty_queue(monkeypatch):
    import subprocess
    monkeypatch.setattr(guard.subprocess, 'run', lambda cmd, **kw:
                        subprocess.CompletedProcess(cmd, 1, '', 'JobServer connection refused'))
    with pytest.raises(guard.SubmissionBlocked, match='Cannot verify'):
        with guard.submission_guard('/suite','mmgbsa'):pytest.fail('Unverified empty queue')
