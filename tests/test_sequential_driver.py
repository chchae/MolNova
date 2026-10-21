import json
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, driver, cli, schrodinger_guard as guard
from molnova.states import CompoundState as State


def project(tmp_path, state=State.GENERATED, iteration=3):
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        conn.execute(
            "INSERT INTO compound(name,smiles,iteration,state,docking_score,prime_gbsa_score) "
            "VALUES ('CMP','C',?,?,?,?)",
            (iteration, state, -10 if state in {State.DOCKED, State.PRIME_GBSA_DONE} else None,
             -20 if state == State.PRIME_GBSA_DONE else None),
        )
    return SimpleNamespace(db_path=db, output=tmp_path, target_count=1,
                           prime_gbsa_input_count=1, prime_gbsa_elite_count=10, max_iteration=5,
                           synthetic_feasibility_enabled=True, schrodinger=Path('/suite'))


def test_sequential_workers_share_iteration_and_wait_for_real_completion(tmp_path, monkeypatch):
    args = project(tmp_path)
    calls = []
    waits = []
    transitions = {
        'generate': "UPDATE compound SET state='generated'",
        'druglike': 'SELECT 1',
        'synthetic': "UPDATE compound SET synthetic_feasibility=1,sa_score=2",
        'ligprep': "UPDATE compound SET state='ligprepped'",
        'glide': "UPDATE compound SET state='docked',docking_score=-10",
        'prime_gbsa': "UPDATE compound SET state='prime_gbsa_done',prime_gbsa_score=-20",
    }

    class Stop(driver.StopFlag):
        def wait(self, seconds):
            waits.append(seconds)

    def process(stage, cmd, cwd, stop):
        assert cmd[-2:] == ['--iteration', '3']
        assert cwd == tmp_path
        calls.append(stage)
        # Exit 0 is not a completed calculation. Glide and GBSA need another pass.
        if stage in {'glide', 'prime_gbsa'} and calls.count(stage) == 1:
            return 0
        with database.connect(args.db_path) as conn:
            conn.execute(transitions[stage])
        return 0

    monkeypatch.setattr(driver, 'stream_process', process)
    monkeypatch.setattr(guard, 'active_jobs', lambda suite: [])
    assert driver.supervise_iteration(args, tmp_path/'project.toml', 3,
                                     driver.build_stages(True, False), 30, Stop())
    assert calls == ['generate', 'druglike', 'synthetic', 'ligprep', 'glide', 'glide', 'prime_gbsa', 'prime_gbsa']
    assert waits == [30, 5]
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT DISTINCT iteration FROM compound').fetchall() == [(3,)]


@pytest.mark.parametrize('rc', [0, 1])
def test_once_stops_at_incomplete_or_failed_stage(tmp_path, monkeypatch, rc):
    args = project(tmp_path)
    args.target_count = 2
    calls = []
    monkeypatch.setattr(driver, 'stream_process', lambda stage, *a: calls.append(stage) or rc)
    assert not driver.supervise_iteration(args, tmp_path/'p.toml', 3,
        driver.build_stages(True, False), 30, driver.StopFlag(), once=True)
    assert calls == ['generate']


def test_retry_failure_cannot_advance_stage(tmp_path, monkeypatch):
    args = project(tmp_path)
    calls = []
    class Stop(driver.StopFlag):
        def wait(self, seconds):
            self.set()
    monkeypatch.setattr(driver, 'stream_process', lambda stage, *a: calls.append(stage) or 1)
    assert not driver.supervise_iteration(args, tmp_path/'p.toml', 3,
        driver.build_stages(True, False), 30, Stop())
    assert calls == ['generate']


def test_restart_resumes_oldest_unfinished_iteration_and_never_requires_elites(tmp_path):
    args = project(tmp_path, State.PRIME_GBSA_DONE, iteration=1)
    assert driver.choose_iteration(args) == 2
    with database.connect(args.db_path) as conn:
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('later','C',3,'generated')")
    assert driver.choose_iteration(args) == 3
    assert driver.choose_iteration(args, 1) == 1
    args.max_iteration = 1
    with database.connect(args.db_path) as conn:
        conn.execute('DELETE FROM compound WHERE iteration=3')
    assert driver.choose_iteration(args) is None
    with pytest.raises(ValueError):
        driver.choose_iteration(args, -1)


def test_terminal_failures_are_processed_and_top_n_excludes_others(tmp_path):
    args = project(tmp_path, State.FAILED)
    assert database.stage_completion_reason(args.db_path, 3, 'ligprep', 1, 1, True) is None
    assert database.stage_completion_reason(args.db_path, 3, 'prime_gbsa', 1, 1) is None
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='prime_gbsa_done',docking_score=-10,prime_gbsa_score=-20")
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,docking_score) "
                     "VALUES ('outside','C',3,'docked',-5)")
    assert database.stage_completion_reason(args.db_path, 3, 'prime_gbsa', 2, 1) is None


def test_later_iteration_cannot_mask_requested_iteration(tmp_path):
    args = project(tmp_path)
    with database.connect(args.db_path) as conn:
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,prime_gbsa_score,docking_score) "
                     "VALUES ('later','C',4,'prime_gbsa_done',-30,-15)")
    assert database.stage_completion_reason(args.db_path, 3, 'prime_gbsa', 1, 1)


def test_live_native_job_overrides_completed_db_and_zero_exit(tmp_path, monkeypatch):
    args = project(tmp_path, State.DOCKED)
    live = {'jobId': 'child', 'status': 'RUNNING', 'spec': {'launchParams': {
        'launchDirectory': str(tmp_path/'iter3/glide/ref_6')}}}
    monkeypatch.setattr(guard, 'active_jobs', lambda suite: [live])
    monkeypatch.setattr(driver, 'stream_process', lambda *a: 0)
    assert not driver.supervise_iteration(args, tmp_path/'p.toml', 3,
        [('glide','module'), ('prime_gbsa','module')], 30, driver.StopFlag(), once=True)
    live['spec']['launchParams']['launchDirectory'] = str(tmp_path/'iter4/glide/ref_6')
    assert driver.external_completion_reason(args, 3, 'glide') is None


def test_jobserver_lookup_failure_blocks_transition(tmp_path, monkeypatch):
    args = project(tmp_path)
    def fail(suite):
        raise OSError('unreachable')
    monkeypatch.setattr(guard, 'active_jobs', fail)
    assert 'cannot verify' in driver.external_completion_reason(args, 3, 'glide')


def test_uncertain_glide_record_and_unretired_prime_queue_block_transition(tmp_path):
    args = project(tmp_path)
    directory = tmp_path/'iter3/glide/ref_6'
    directory.mkdir(parents=True)
    (directory/'glide_job.json').write_text(json.dumps({'job_id': None}))
    assert 'uncertain' in driver.external_completion_reason(args, 3, 'glide')
    directory = tmp_path/'iter3/prime_gbsa'
    directory.mkdir(parents=True)
    (directory/'prime_gbsa_job.json').write_text(json.dumps({'mode':'compound_queue_v1'}))
    assert 'recovery' in driver.external_completion_reason(args, 3, 'prime_gbsa')


def test_cli_forwards_iteration_to_driver(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'driver_main', lambda argv: calls.append(argv))
    cli.main(['run', str(tmp_path/'p.toml'), '--iteration', '3'])
    assert calls == [[str(tmp_path/'p.toml'), '--poll-interval', '30', '--iteration', '3']]


def test_silent_worker_shutdown_is_responsive(tmp_path):
    class Stop:
        def is_set(self):
            return time.monotonic() - started > 0.2
    started = time.monotonic()
    assert driver.stream_process('glide', [sys.executable, '-c',
        'import time; time.sleep(60)'], tmp_path, Stop()) != 0
    assert time.monotonic() - started < 3


def test_real_subprocess_stages_run_in_order_and_exit_after_one_iteration(tmp_path, monkeypatch):
    args = project(tmp_path)
    args.synthetic_feasibility_enabled = False
    module = tmp_path/'fake_worker.py'
    module.write_text(r"""import argparse, sqlite3
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('project'); p.add_argument('--iteration', type=int)
a=p.parse_args(); root=Path(a.project).parent
with sqlite3.connect(root/'project.sqlite') as c:
    state=c.execute('SELECT state FROM compound WHERE iteration=?',(a.iteration,)).fetchone()[0]
    transitions={'generated':('ligprepped',None,None),'ligprepped':('docked',-10,None),'docked':('prime_gbsa_done',-10,-20)}
    new,dock,prime_gbsa=transitions[state]
    c.execute('UPDATE compound SET state=?,docking_score=?,prime_gbsa_score=? WHERE iteration=?',(new,dock,prime_gbsa,a.iteration))
with (root/'order.log').open('a') as f:f.write(str(a.iteration)+':'+new+'\n')
""")
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    monkeypatch.setattr(guard, 'active_jobs', lambda suite: [])
    stages = [(stage, 'fake_worker') for stage in ('ligprep','glide','prime_gbsa')]
    class Deadline(driver.StopFlag):
        def is_set(self):
            return time.monotonic() - started > 5
    started = time.monotonic()
    assert driver.supervise_iteration(args, tmp_path/'p.toml', 3, stages, 1, Deadline())
    assert (tmp_path/'order.log').read_text().splitlines() == ['3:ligprepped','3:docked','3:prime_gbsa_done']


def test_saved_glide_child_blocks_even_when_list_is_empty(tmp_path, monkeypatch):
    args = project(tmp_path, State.DOCKED)
    directory = tmp_path/'iter3/glide/ref_6'; directory.mkdir(parents=True)
    (directory/'glide_job.json').write_text(json.dumps({'job_id':'parent', 'downloaded':False}))
    monkeypatch.setattr(guard, 'active_jobs', lambda *a: [])
    monkeypatch.setattr(guard, 'job_details', lambda *a: [
        {'jobId':'parent', 'status':'DONE'}, {'jobId':'child','status':'RUNNING'}])
    assert 'still active' in driver.external_completion_reason(args, 3, 'glide')


@pytest.mark.parametrize('status,removed', [('RUNNING',False), ('DONE',True)])
def test_ligprep_postcommit_restart_retires_only_terminal_record(tmp_path, monkeypatch, status, removed):
    from molnova.stages import ligprep
    from molnova.ligprep_recovery import _save, load_job
    args = project(tmp_path, State.LIGPREPPED)
    args.synthetic_feasibility_enabled = False
    directory = tmp_path/'iter3/ligprep'; directory.mkdir(parents=True)
    _save(directory, {'job_id':'parent', 'compound_ids':[1]})
    monkeypatch.setattr(guard, 'job_details', lambda *a: [
        {'jobId':'parent','status':'DONE'}, {'jobId':'child','status':status}])
    ligprep._run_stage(args, 3)
    assert (load_job(directory) is None) == removed


def test_fep_export_is_a_single_final_stage(tmp_path, monkeypatch):
    args = project(tmp_path, State.PRIME_GBSA_DONE)
    calls = []
    monkeypatch.setattr(driver, 'stream_process', lambda stage,*a: calls.append(stage) or 0)
    assert driver.supervise_iteration(args, tmp_path/'p.toml', 3,
        [('fep','module')], 30, driver.StopFlag())
    assert calls == ['fep']



def test_iteration_timing_survives_restart_and_completion_is_idempotent(tmp_path, monkeypatch):
    args = project(tmp_path)
    with database.connect(args.db_path) as conn:
        conn.execute('DELETE FROM compound')
    monkeypatch.setattr(database.time, 'time', lambda: 1000)
    assert database.start_iteration_timer(args.db_path, 3) == 1000
    monkeypatch.setattr(database.time, 'time', lambda: 2000)
    assert database.start_iteration_timer(args.db_path, 3) == 1000
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT finished_at FROM iteration_timing').fetchone() == (None,)
    monkeypatch.setattr(database.time, 'time', lambda: 4600)
    assert database.finish_iteration_timer(args.db_path, 3) == (3600, False)
    monkeypatch.setattr(database.time, 'time', lambda: 5000)
    assert database.start_iteration_timer(args.db_path, 3) == 1000
    assert database.finish_iteration_timer(args.db_path, 3) == (3600, False)


def test_legacy_iteration_timing_is_labelled_estimated(tmp_path, monkeypatch):
    args = project(tmp_path, State.PRIME_GBSA_DONE)
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET created_at='1970-01-01 00:10:00'")
    monkeypatch.setattr(database.time, 'time', lambda: 1800)
    assert database.start_iteration_timer(args.db_path, 3) == 600
    assert database.finish_iteration_timer(args.db_path, 3) == (1200, True)
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT state,prime_gbsa_score FROM compound').fetchone() == ('prime_gbsa_done', -20)


@pytest.mark.parametrize('completed', [True, False])
def test_driver_reports_total_time_only_after_verified_completion(tmp_path, monkeypatch, capsys, completed):
    args = project(tmp_path, State.PRIME_GBSA_DONE)
    args.project = 'project'
    args.fep_enabled = False
    with database.connect(args.db_path) as conn:
        conn.execute('DELETE FROM compound')
    monkeypatch.setattr(driver.c, 'configure_project', lambda _: args)
    monkeypatch.setattr(driver, 'choose_iteration', lambda *a: 3)
    monkeypatch.setattr(driver.signal, 'signal', lambda *a: None)
    monkeypatch.setattr(database.time, 'time', lambda: 1000)
    def supervise(*a):
        monkeypatch.setattr(database.time, 'time', lambda: 4661)
        return completed
    monkeypatch.setattr(driver, 'supervise_iteration', supervise)
    driver.main([str(tmp_path / 'project.toml')])
    output = capsys.readouterr().out
    with database.connect(args.db_path) as conn:
        finished = conn.execute('SELECT finished_at FROM iteration_timing').fetchone()[0]
    if completed:
        assert 'iteration 3: finished; total elapsed=01:01:01.0.' in output
        assert finished == 4661
    else:
        assert 'paused; restart to resume' in output
        assert 'total elapsed=' not in output
        assert finished is None


@pytest.mark.parametrize('initial_stages', ['prime-fep', 'prime', 'fep'])
def test_enabling_md_at_runtime_supervises_it_before_fep_or_exit(tmp_path, monkeypatch, initial_stages):
    args = project(tmp_path, state=State.PRIME_GBSA_DONE)
    args.md_gbsa_enabled = False
    config = tmp_path/'project.toml'
    config.write_text('md-gbsa = '+('true' if initial_stages=='fep' else 'false')+'\n')
    stages = []
    if initial_stages.startswith('prime'):
        stages.append(('prime_gbsa','molnova.stages.prime_gbsa'))
    if 'fep' in initial_stages:
        stages.append(('fep','molnova.stages.fep'))
    calls=[]
    class Stop(driver.StopFlag):
        def wait(self, seconds):
            pass
    def process(stage, cmd, cwd, stop):
        calls.append(stage)
        if stage=='prime_gbsa':
            config.write_text('md-gbsa = true\n')
        elif stage=='md-gbsa' and calls.count('md-gbsa')==2:
            database.freeze_md_gbsa_selection(args.db_path,3,
                dict(protocol=dict(snapshots=20),replicates=1),count=1,target_count=1,prime_count=1)
            database.store_md_gbsa_result(args.db_path,3,1,dict(id=1,mean=-40.,std=0.,
                snapshots=[dict(frame=i,score=-40.) for i in range(20)]))
        elif stage=='fep':
            assert database.md_gbsa_completion_reason(args.db_path,3) is None
        return 0
    monkeypatch.setattr(driver,'stream_process',process)
    monkeypatch.setattr(guard,'active_jobs',lambda suite:[])
    assert driver.supervise_iteration(args,config,3,stages,30,Stop())
    assert calls==(['prime_gbsa'] if initial_stages.startswith('prime') else [])+['md-gbsa','md-gbsa']+(['fep'] if 'fep' in initial_stages else [])


def test_disabling_unsubmitted_md_skips_stage_but_keeps_saved_ownership(tmp_path):
    args=project(tmp_path,state=State.PRIME_GBSA_DONE)
    args.md_gbsa_enabled=True
    config=tmp_path/'project.toml';config.write_text('md-gbsa = false\n')
    assert not driver.md_stage_required(args,config,3)
    database.freeze_md_gbsa_selection(args.db_path,3,
        dict(protocol=dict(snapshots=20),replicates=1),count=1,target_count=1,prime_count=1)
    assert driver.md_stage_required(args,config,3)


def test_md_enabled_during_fep_poll_returns_to_prerequisite(tmp_path, monkeypatch):
    args=project(tmp_path,state=State.PRIME_GBSA_DONE)
    config=tmp_path/'project.toml';config.write_text('md-gbsa = false\n')
    calls=[]
    def process(stage,*unused):
        calls.append(stage)
        if stage=='fep':
            config.write_text('md-gbsa = true\n')
            return 1
        # Stop at the MD prerequisite without submitting scientific work.
        return 0
    monkeypatch.setattr(driver,'stream_process',process)
    class Stop(driver.StopFlag):
        def wait(self,seconds):
            self.set()
    assert not driver.supervise_iteration(args,config,3,[('fep','molnova.stages.fep')],30,Stop())
    assert calls==['fep','md-gbsa']
