import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, ligprep_recovery as recovery, prime_async
from molnova.stages import ligprep, glide, mmgbsa
from molnova.states import CompoundState as State


def project(tmp_path, states):
    db = tmp_path / 'project.sqlite'
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany('INSERT INTO compound(name,smiles,iteration,state,docking_score) '
                         "VALUES (?,'C',1,?,?)",
                         [(str(i), state, -10+i if state in (State.DOCKED, State.GBSA_RUNNING) else None)
                          for i, state in enumerate(states)])
    return SimpleNamespace(db_path=db, output=tmp_path, target_count=2,
                           synthetic_feasibility_enabled=False, gbsa_input_count=200)


def test_generation_barrier_blocks_partial_iteration_and_live_generator(tmp_path, monkeypatch):
    args = project(tmp_path, [State.GENERATED])
    monkeypatch.setattr(_core, 'run_ligprep_once', lambda **kw: pytest.fail('early LigPrep'))
    ligprep._run_stage(args, 1)
    args.target_count = 1
    with database.stage_lock(args.db_path, 'generate'):
        ligprep._run_stage(args, 1)
    assert database.upstream_completion_reason(args.db_path, 1, 1, 'ligprep') is None


@pytest.mark.parametrize('state', [State.GENERATED, State.SYNTHETIC_RUNNING, State.LIGPREP_RUNNING])
def test_glide_barrier_blocks_every_unfinished_ligprep_state(tmp_path, monkeypatch, state):
    args = project(tmp_path, [State.LIGPREPPED, state])
    monkeypatch.setattr(_core, 'assign_reference_compounds', lambda *a: pytest.fail('early Glide'))
    glide._run_stage(args, SimpleNamespace(iteration=1, poll_interval=1,
                                          completion_fraction=.95, tail_timeout=1))


def test_glide_waits_until_ligprep_worker_exits(tmp_path):
    args = project(tmp_path, [State.LIGPREPPED, State.FAILED])
    with database.stage_lock(args.db_path, 'ligprep'):
        assert 'worker still running' in database.upstream_completion_reason(args.db_path, 1, 2, 'glide')
    assert database.upstream_completion_reason(args.db_path, 1, 2, 'glide') is None


def test_concurrent_gbsa_claims_are_disjoint_and_top_200_is_fixed(tmp_path):
    args = project(tmp_path, [State.DOCKED]*205)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: database.claim_mmgbsa_candidates(args.db_path, 1, 200, 205), range(2)))
    assert sorted(map(len, results)) == [0, 200]
    assert set(results[0]).isdisjoint(results[1])
    assert database.mmgbsa_candidates(args.db_path, 1, 200, 205) == []


def test_restart_resets_only_claims_without_saved_submission(tmp_path, monkeypatch):
    args = project(tmp_path, [State.GBSA_RUNNING, State.GBSA_RUNNING])
    directory = tmp_path / 'iter1/mmgbsa'; directory.mkdir(parents=True)
    prime_async._save(directory, {'command': ['/suite/prime_mmgbsa'], 'compound_ids': [1], 'job_id': 'live'})
    calls = []
    monkeypatch.setattr(_core, 'run_iteration_mmgbsa', lambda **kw: calls.append(kw['compound_ids']))
    mmgbsa._run_stage(args, 1)
    assert calls == [[1]]
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute('SELECT state FROM compound ORDER BY id').fetchall() == [('gbsa_running',), ('docked',)]
    assert database.claim_mmgbsa_candidates(args.db_path, 1, 200, 2) == [2]


def test_ligprep_restart_polls_live_job_then_downloads_completed_output(tmp_path, monkeypatch):
    output = tmp_path / 'ligprep_all.maegz'
    recovery._save(tmp_path, {'command': ['/suite/ligprep'], 'compound_ids': [1], 'job_id': 'live'})
    calls = []
    status = 'RUNNING'
    def query(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == 'download':
            output.write_text('prepared')
            return SimpleNamespace(stdout='')
        return SimpleNamespace(stdout=json.dumps({'jobId':'child','status':'DONE'})+'\n'+
                               json.dumps({'jobId':'live','status':status}))
    monkeypatch.setattr(recovery.subprocess, 'run', query)
    monkeypatch.setattr(recovery.subprocess, 'Popen', lambda *a, **kw: pytest.fail('duplicate LigPrep'))
    assert recovery.run_or_recover([], tmp_path, [1], output) is False
    assert len(calls) == 1
    status = 'DONE'
    assert recovery.run_or_recover([], tmp_path, [1], output) is True
    assert calls[-1] == ['/suite/jsc', 'download', '--cwd', 'live']
    assert recovery.load_job(tmp_path) is not None  # DB commit must precede deletion.


@pytest.mark.parametrize('status', ['FAILED', 'CANCELED', 'STOPPED'])
def test_ligprep_terminal_failure_allows_safe_retry(tmp_path, monkeypatch, status):
    recovery._save(tmp_path, {'command': ['/suite/ligprep'], 'compound_ids':[1], 'job_id':'old'})
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps({'jobId':'old','status':status})))
    with pytest.raises(RuntimeError, match='ended with'):
        recovery.run_or_recover([], tmp_path, [1], tmp_path/'ligprep_all.maegz')
    assert recovery.load_job(tmp_path) is None


def test_ligprep_uncertain_launch_and_status_failure_never_reset_or_resubmit(tmp_path, monkeypatch):
    recovery._save(tmp_path, {'command':['/suite/ligprep'], 'compound_ids':[1], 'job_id':None})
    monkeypatch.setattr(recovery.subprocess, 'Popen', lambda *a, **kw: pytest.fail('duplicate'))
    with pytest.raises(RuntimeError, match='without saved JobId'):
        recovery.run_or_recover([], tmp_path, [1], tmp_path/'output')
    assert recovery.load_job(tmp_path) is not None


def test_new_ligprep_submission_saves_job_id_before_wait_finishes(tmp_path, monkeypatch):
    output = tmp_path/'ligprep_all.maegz'
    class Process:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        @property
        def stdout(self):
            yield 'JobId: job-1\n'
            assert recovery.load_job(tmp_path)['job_id'] == 'job-1'
            output.write_text('complete')
        def wait(self): return 0
    monkeypatch.setattr(recovery.subprocess, 'Popen', lambda *a, **kw: Process())
    assert recovery.run_or_recover(['/suite/ligprep', '-WAIT'], tmp_path, [1], output)


def test_gbsa_recovers_job_record_after_scores_committed(tmp_path, monkeypatch):
    args = project(tmp_path, [State.DOCKED, State.DOCKED])
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='gbsa_done',gbsa_score=-40")
    directory = tmp_path/'iter1/mmgbsa'; directory.mkdir(parents=True)
    prime_async._save(directory, {'command':['/suite/prime_mmgbsa'], 'compound_ids':[1,2], 'job_id':'done'})
    def recover(**kw):
        assert kw['compound_ids'] == [1,2]
        prime_async.job_file(directory).unlink()
        return 0
    monkeypatch.setattr(_core, 'run_iteration_mmgbsa', recover)
    mmgbsa._run_stage(args, 1)
    assert prime_async.load_job(directory) is None
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute('SELECT state,gbsa_score FROM compound').fetchall() == [('gbsa_done',-40),('gbsa_done',-40)]


def test_ligprep_status_failure_retains_record(tmp_path, monkeypatch):
    recovery._save(tmp_path, {'command':['/suite/ligprep'], 'compound_ids':[1], 'job_id':'live'})
    def offline(*a, **kw):
        raise recovery.subprocess.CalledProcessError(1, 'jsc')
    monkeypatch.setattr(recovery.subprocess, 'run', offline)
    with pytest.raises(recovery.subprocess.CalledProcessError):
        recovery.run_or_recover([], tmp_path, [1], tmp_path/'output')
    assert recovery.load_job(tmp_path)['job_id'] == 'live'


def test_legacy_ligprep_job_is_polled_without_resubmission(tmp_path, monkeypatch):
    (tmp_path/'ligprep_all.log').write_text('JobId: legacy\n')
    monkeypatch.setattr(recovery.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps({'jobId':'legacy','status':'RUNNING'})))
    monkeypatch.setattr(recovery.subprocess, 'Popen', lambda *a, **kw: pytest.fail('legacy job duplicate'))
    assert recovery.run_or_recover(['/suite/ligprep'], tmp_path, [1], tmp_path/'output') is False
    assert recovery.load_job(tmp_path)['job_id'] == 'legacy'
