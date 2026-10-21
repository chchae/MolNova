import json
from types import SimpleNamespace

import pytest

from molnova import _core, database
from molnova.stages._summary import completion_summary


@pytest.fixture
def project(tmp_path):
    args = SimpleNamespace(db_path=tmp_path/'project.sqlite', output=tmp_path)
    with database.connect(args.db_path) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany(
            'INSERT INTO compound(id,name,smiles,iteration,state,failed_stage,docking_score,amber_gbsa_score,primg_gbsa_score) '
            "VALUES (?,?,'CC',1,?,?,?,?,?)",
            [(1,'one','primg_gbsa_done',None,-10,-20,-30),
             (2,'two','failed','glide',None,None,None),
             (3,'three','failed','primg_gbsa',-9,-19,None),
             (4,'four','sa_rejected',None,None,None,None),
             (5,'five','failed','ligprep',None,None,None)])
    return args


@pytest.mark.parametrize('stage,successes,failures', [
    ('generate',{1,2,3,5},set()), ('ligprep',{1,2,3},{5}),
    ('glide',{1,3},{2}), ('amber-gbsa',{1,3},set()), ('primg_gbsa',{1},{3}),
])
def test_summary_counts_original_compounds_and_attributes_failures(project, stage, successes, failures):
    assert database.stage_summary_results(project.db_path,1,stage) == (successes,failures)
    summary = completion_summary(project,1,stage,120)
    n = len(successes) + len(failures)
    assert f'compounds={n} (ok={len(successes)}, failed={len(failures)})' in summary
    assert f'wall/compound={120/n:.2f}s' in summary


def test_native_prime_times_use_only_completed_scored_ids_and_do_not_double_count(project):
    root = project.output/'iter1/primg_gbsa'
    root.mkdir(parents=True)
    for run in ['queue-1','queue-2']:
        (root/f'{run}.completed.json').write_text(json.dumps({
            'mode':'compound_queue_v1', 'run_dir':run,
            'entries':{'1':'done','2':'failed','3':'done'}}))
        for cid in (1,2,3):
            folder=root/run/f'CMPID_{cid}'
            folder.mkdir(parents=True)
            (folder/'primg_gbsa.log').write_text(
                f'Total Time Elapsed (hours:minutes:seconds) 0:00:{30 if cid==1 else 99}\n')
    (root/'queue-3.completed.json').write_text('{broken')
    summary=completion_summary(project,1,'primg_gbsa',120)
    assert 'native job n=1/1 avg/median/min/max=30.0/30.0/30.0/30.0s' in summary
    assert 'wall/compound=60.00s' in summary


def test_missing_prime_logs_are_unavailable_not_zero(project):
    assert 'native job time=n/a' in completion_summary(project,1,'primg_gbsa',120)
    assert 'wall/compound=n/a' in completion_summary(project,99,'primg_gbsa',120)


def test_amber_failure_preserving_prior_prime_result_is_still_counted(project):
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET amber_gbsa_score=NULL,failed_stage='amber_gbsa' WHERE id=1")
    assert database.stage_summary_results(project.db_path,1,'amber-gbsa') == ({3},{1})


def test_native_amber_timings_support_old_and_new_remote_results(project):
    root=project.output/'iter1/amber_gbsa'
    root.mkdir(parents=True)
    snapshot=root/'remote_snapshot.json'
    snapshot.write_text(json.dumps({'results':[{'id':1,'score':-20}, {'id':3,'score':-19}]}))
    assert 'native job time=n/a' in completion_summary(project,1,'amber-gbsa',120)
    snapshot.write_text(json.dumps({'results':[{'id':1,'score':-20,'elapsed_s':40},
                                             {'id':3,'score':-19,'elapsed_s':60},
                                             {'id':2,'error':'failed','elapsed_s':999}]}))
    assert 'native job n=2/2 avg/median/min/max=50.0/50.0/40.0/60.0s' in completion_summary(project,1,'amber-gbsa',120)


def test_missing_and_invalid_durations_show_coverage(project):
    root=project.output/'iter1/amber_gbsa'
    root.mkdir(parents=True)
    (root/'remote_snapshot.json').write_text(json.dumps({'results':[
        {'id':1,'score':-20,'elapsed_s':12}, {'id':3,'score':-19,'elapsed_s':float('nan')}]}))
    assert 'native job n=1/2' in completion_summary(project,1,'amber-gbsa',120)


@pytest.mark.parametrize('snapshot', [None, [], {'results':None}, {'results':[None, {'id':[], 'elapsed_s':20}]}])
def test_damaged_diagnostic_artifacts_do_not_break_completion(project, snapshot):
    root=project.output/'iter1/amber_gbsa'
    root.mkdir(parents=True)
    (root/'remote_snapshot.json').write_text(json.dumps(snapshot))
    assert 'native job time=n/a' in completion_summary(project,1,'amber-gbsa',120)
