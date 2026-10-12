from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, driver


def project(tmp_path):
    db = tmp_path/'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,docking_score,amber_gbsa_score,gbsa_score) "
                     "VALUES ('CMP','CCO',1,'gbsa_done',-10,-30,-40)")
    return SimpleNamespace(db_path=db, output=tmp_path, target_count=1, gbsa_input_count=1,
                           synthetic_feasibility_enabled=False, amber_gbsa_enabled=True,
                           amber_keep_count=500, schrodinger=Path('/suite'))


@pytest.mark.parametrize('stage,label', [('ligprep','LigPrep'), ('glide','Glide docking'),
                                       ('amber-gbsa','Amber MM-GBSA'), ('mmgbsa','Prime MM-GBSA')])
def test_each_calculation_stage_reports_total_after_verified_completion(tmp_path, monkeypatch, capsys, stage, label):
    args = project(tmp_path)
    clock = [1000.0]
    monkeypatch.setattr(database.time,'time',lambda:clock[0])
    def process(*a):
        clock[0] = 4661.5
        return 0
    monkeypatch.setattr(driver,'stream_process',process)
    monkeypatch.setattr(driver,'external_completion_reason',lambda *a:None)
    assert driver.supervise_iteration(args,tmp_path/'p.toml',1,[(stage,'module')],30,driver.StopFlag())
    assert f'{label} stage complete; total elapsed=01:01:01.5' in capsys.readouterr().out
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT iteration,stage,started_at,finished_at FROM stage_timing').fetchone() == (1,stage,1000,4661.5)
        assert conn.execute('SELECT state,docking_score,amber_gbsa_score,gbsa_score FROM compound').fetchone() == ('gbsa_done',-10,-30,-40)


@pytest.mark.parametrize('blocked', ['worker_failure','db_pending','external_pending','stop'])
def test_paused_stage_keeps_start_and_finishes_only_after_restart(tmp_path, monkeypatch, capsys, blocked):
    args = project(tmp_path)
    clock = [1000.0]
    monkeypatch.setattr(database.time,'time',lambda:clock[0])
    stop = driver.StopFlag()
    if blocked == 'db_pending':
        with database.connect(args.db_path) as conn:
            conn.execute("UPDATE compound SET state='glide_running',docking_score=NULL")
    def process(*a):
        clock[0] = 1060
        if blocked == 'stop':
            stop.set()
        return 1 if blocked == 'worker_failure' else 0
    monkeypatch.setattr(driver,'stream_process',process)
    monkeypatch.setattr(driver,'external_completion_reason',lambda *a:'native child active' if blocked == 'external_pending' else None)
    assert not driver.supervise_iteration(args,tmp_path/'p.toml',1,[('glide','module')],30,stop,once=True)
    assert 'total elapsed=' not in capsys.readouterr().out
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT started_at,finished_at FROM stage_timing').fetchone() == (1000,None)
        conn.execute("UPDATE compound SET state='gbsa_done',docking_score=-10")
    # A new supervisor pass must include the first pass and intervening downtime.
    clock[0] = 4000
    def resumed(*a):
        clock[0] = 4661.5
        return 0
    monkeypatch.setattr(driver,'stream_process',resumed)
    monkeypatch.setattr(driver,'external_completion_reason',lambda *a:None)
    assert driver.supervise_iteration(args,tmp_path/'p.toml',1,[('glide','module')],30,driver.StopFlag())
    assert 'Glide docking stage complete; total elapsed=01:01:01.5' in capsys.readouterr().out


def test_stage_timing_is_per_iteration_and_preserves_first_completion(tmp_path, monkeypatch):
    args = project(tmp_path)
    monkeypatch.setattr(database.time,'time',lambda:1000)
    database.start_stage_timer(args.db_path,1,'glide')
    monkeypatch.setattr(database.time,'time',lambda:2000)
    database.start_stage_timer(args.db_path,2,'glide')
    database.start_stage_timer(args.db_path,1,'amber-gbsa')
    assert database.finish_stage_timer(args.db_path,1,'glide') == 1000
    monkeypatch.setattr(database.time,'time',lambda:5000)
    database.start_stage_timer(args.db_path,1,'glide')
    assert database.finish_stage_timer(args.db_path,1,'glide') == 1000
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT iteration,stage,started_at,finished_at FROM stage_timing ORDER BY iteration,stage').fetchall() == [
            (1,'amber-gbsa',2000,None), (1,'glide',1000,2000), (2,'glide',2000,None)]


def test_polling_waits_are_included_in_full_stage_total(tmp_path, monkeypatch, capsys):
    args = project(tmp_path)
    clock = [1000.0]
    monkeypatch.setattr(database.time,'time',lambda:clock[0])
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='docked',gbsa_score=NULL")
    passes = []
    def process(*a):
        passes.append(1)
        clock[0] += 10
        if len(passes) == 2:
            with database.connect(args.db_path) as conn:
                conn.execute("UPDATE compound SET state='gbsa_done',gbsa_score=-40")
        return 0
    class Stop(driver.StopFlag):
        def wait(self, seconds):
            clock[0] += seconds
    monkeypatch.setattr(driver,'stream_process',process)
    monkeypatch.setattr(driver,'external_completion_reason',lambda *a:None)
    assert driver.supervise_iteration(args,tmp_path/'p.toml',1,[('mmgbsa','module')],30,Stop())
    assert len(passes) == 2
    output = capsys.readouterr().out
    assert output.count('Prime MM-GBSA stage complete') == 1
    assert 'total elapsed=00:00:25.0' in output


def test_prime_recovery_starts_timer_without_finishing_final_prime_stage(tmp_path, monkeypatch, capsys):
    args = project(tmp_path)
    clock = [1000.0]
    monkeypatch.setattr(database.time,'time',lambda:clock[0])
    monkeypatch.setattr(driver,'stream_process',lambda *a:0)
    monkeypatch.setattr(driver,'external_completion_reason',lambda *a:None)
    assert driver.supervise_iteration(args,tmp_path/'p.toml',1,[('mmgbsa-recovery','module')],30,driver.StopFlag())
    assert 'Prime MM-GBSA stage complete' not in capsys.readouterr().out
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT stage,started_at,finished_at FROM stage_timing').fetchone() == ('mmgbsa',1000,None)
    clock[0] = 4000
    assert driver.supervise_iteration(args,tmp_path/'p.toml',1,[('mmgbsa','module')],30,driver.StopFlag())
    assert 'Prime MM-GBSA stage complete; total elapsed=00:50:00.0' in capsys.readouterr().out


def test_already_stopped_supervisor_does_not_start_stage_timer(tmp_path, monkeypatch):
    args = project(tmp_path)
    stop = driver.StopFlag()
    stop.set()
    monkeypatch.setattr(driver,'stream_process',lambda *a:pytest.fail('Stopped worker launched'))
    assert not driver.supervise_iteration(args,tmp_path/'p.toml',1,[('glide','module')],30,stop)
    with database.connect(args.db_path) as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='stage_timing'").fetchone()
