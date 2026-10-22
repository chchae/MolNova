"""Glide ranking, disabled calculators and durable recovery; no external science."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, driver, amber_gbsa
from molnova.config import amber_settings, md_gbsa_settings
from molnova.paths import stage_directory
from molnova.stages import prime_gbsa as prime_stage, amber_gbsa as amber_stage
from molnova.states import CompoundState as State


@pytest.fixture
def project(tmp_path):
    db = tmp_path/'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('ref','C',0,'reference')")
        conn.executemany('INSERT INTO compound(name,smiles,iteration,state,docking_score,prime_gbsa_score) '
                         'VALUES (?,\'C\',1,?,?,?)',
                         [('first',State.DOCKED,-10,None), ('tie',State.DOCKED,-10,-1),
                          ('third',State.DOCKED,-9,-100), ('infinite',State.DOCKED,float('inf'),-200)])
    return SimpleNamespace(db_path=db, output=tmp_path,target_count=4,prime_gbsa_input_count=3,
        prime_gbsa_elite_count=2,max_iteration=3,prime_gbsa_enabled=False,amber_gbsa_enabled=False,
        md_gbsa_enabled=False,synthetic_feasibility_enabled=False,fep_enabled=True,
        schrodinger=Path('/suite'),amber_keep_count=4,amber_remote=dict(host='slurm'))


@pytest.mark.parametrize('key', ['amber-gbsa','amber_gbsa','amber-gbsa-enabled'])
def test_amber_alias_and_prime_default(key,tmp_path):
    assert amber_settings({key:True,'out-dir':'output'},tmp_path)['amber_gbsa_enabled']
    assert amber_settings({'out-dir':'output'},tmp_path)['prime_gbsa_enabled']
    assert not amber_settings({'prime-gbsa':False,'out-dir':'output'},tmp_path)['prime_gbsa_enabled']


@pytest.mark.parametrize('key', ['amber-gbsa','prime-gbsa'])
@pytest.mark.parametrize('value', ['false',0,1])
def test_switches_are_strict_booleans(key,value,tmp_path):
    with pytest.raises(ValueError, match='must be a boolean'):
        amber_settings({key:value},tmp_path)


def test_conflicting_aliases_and_md_without_prime_are_rejected(tmp_path):
    with pytest.raises(ValueError,match='must agree'):
        amber_settings({'amber-gbsa':False,'amber_gbsa':True},tmp_path)
    with pytest.raises(ValueError,match='requires prime-gbsa'):
        md_gbsa_settings({'prime-gbsa':False,'md-gbsa':True})


def test_glide_elites_use_finite_scores_ties_and_exact_previous_iteration(project):
    a=project
    assert database.generation_completion_reason(a,1,2) is None
    elites=database.iteration_docking_elites(a.db_path,1,2)
    assert [r[1] for r in elites]==['first','tie']
    assert all(r[3]==1 and r[4]==r[5]==-10 for r in elites)
    assert 'only 3/4 finite Glide' in database.generation_completion_reason(a,1,4)
    with database.connect(a.db_path) as conn:
        assert conn.execute('SELECT prime_gbsa_score FROM compound WHERE name=\'first\'').fetchone()==(None,)
        conn.execute("UPDATE compound SET state='generated',docking_score=NULL WHERE name='third'")
    assert database.generation_completion_reason(a,1,2) is not None


@pytest.mark.parametrize('state', [State.AMBER_RUNNING,State.PRIME_GBSA_RUNNING,State.MD_GBSA_RUNNING])
def test_glide_mode_waits_for_existing_claims(project,state):
    with database.connect(project.db_path) as conn:
        conn.execute('UPDATE compound SET state=? WHERE name=\'first\'',(state,))
    assert database.generation_completion_reason(project,1,2) is not None


def test_driver_skips_unscored_references_and_gbsa_stages(project,monkeypatch):
    monkeypatch.setattr(driver,'external_completion_reason',lambda *a:None)
    assert driver.choose_iteration(project)==2
    assert driver.reference_stages(project)==[]
    stages=[s for s,_ in driver.iteration_stages(project,2)]
    assert stages==['generate','druglike','ligprep','glide','fep']


def test_driver_retains_disabled_saved_jobs_and_completion_after_score_commit(project):
    for stage,file in [('amber_gbsa','amber_job.json'),('prime_gbsa','prime_gbsa_job.json')]:
        folder=stage_directory(project.output,1,stage);folder.mkdir(parents=True,exist_ok=True)
        (folder/file).write_text(json.dumps({'compound_ids':[2]}))
    stages=[s for s,_ in driver.iteration_stages(project,1)]
    assert stages[:2]==['prime_gbsa-recovery','amber-gbsa-recovery']
    assert driver.choose_iteration(project)==1


def test_glide_fep_ignores_existing_prime_scores_and_freezes_selection(project):
    rows=database.fep_candidates(project.db_path,1,2,target_count=4,docking_only=True)
    assert [r['name'] for r in rows]==['first','tie']
    assert all(r['ranking_source']=='docking' for r in rows)
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET docking_score=-100 WHERE name='third'")
    assert database.fep_candidates(project.db_path,1,2,target_count=4,docking_only=True)==rows


def test_disabled_prime_never_claims_new_candidates(project,monkeypatch):
    monkeypatch.setattr(_core,'run_iteration_prime_gbsa',lambda **kw:pytest.fail('New Prime work submitted'))
    prime_stage._run_stage(project,1)
    with database.connect(project.db_path) as conn:
        assert not conn.execute('SELECT 1 FROM compound WHERE state=?',(State.PRIME_GBSA_RUNNING,)).fetchone()


def test_disabled_prime_orphan_claim_is_preserved(project):
    with database.connect(project.db_path) as conn:
        conn.execute('UPDATE compound SET state=? WHERE name=\'first\'',(State.PRIME_GBSA_RUNNING,))
    with pytest.raises(RuntimeError,match='reconcile'):
        prime_stage._run_stage(project,1)
    assert database.has_running_prime_gbsa(project.db_path,1)


def test_disabled_prime_recovers_only_saved_ids(project,monkeypatch):
    a=project
    folder=stage_directory(a.output,1,'prime_gbsa');folder.mkdir(parents=True)
    manifest=folder/'prime_gbsa_job.json'
    active=dict(compound_ids=[2],job_id='saved')
    manifest.write_text(json.dumps(active))
    monkeypatch.setattr(prime_stage,'reconcile_saved_job',lambda *args:active)
    def recover(**kw):
        assert kw['compound_ids']==[2]
        with database.connect(a.db_path) as conn:
            conn.execute('UPDATE compound SET state=?,prime_gbsa_score=-30 WHERE id=2',(State.PRIME_GBSA_DONE,))
        manifest.unlink()
        return 1
    monkeypatch.setattr(_core,'run_iteration_prime_gbsa',recover)
    prime_stage._run_stage(a,1)
    with database.connect(a.db_path) as conn:
        assert conn.execute('SELECT state,prime_gbsa_score FROM compound WHERE id=2').fetchone()==(State.PRIME_GBSA_DONE,-30)
        assert conn.execute('SELECT state FROM compound WHERE id=3').fetchone()==(State.DOCKED,)


def test_disabled_amber_recovers_existing_manifest_without_refilling(project,monkeypatch):
    a=project
    folder=stage_directory(a.output,1,'amber_gbsa');folder.mkdir(parents=True)
    manifest=folder/'amber_job.json'
    manifest.write_text(json.dumps(dict(compound_ids=[2],settings=dict(host='slurm'))))
    monkeypatch.setattr(database,'claim_amber_candidates',lambda *args:pytest.fail('New Amber claims'))
    def recover(args,iteration,directory,record):
        assert record['compound_ids']==[2]
        manifest.unlink()
        return True
    monkeypatch.setattr(amber_gbsa,'poll',recover)
    amber_stage._run_stage(a,1)
    assert not manifest.exists()


def test_glide_training_metadata_does_not_label_docking_scores_as_prime(tmp_path,monkeypatch):
    monkeypatch.setattr(_core,'decompose_elite_for_libinvent',lambda *a:('C[*]','C*'))
    _core.write_elite_libinvent_training_files([(1,'elite','CC',1,-10,-10)],'C[*]',tmp_path,score_source='glide')
    header=(tmp_path/'elite_selected.tsv').read_text().splitlines()[0].split('\t')
    assert 'glide_score' in header and 'prime_gbsa_score' not in header
    assert len(header)==len(set(header))


def test_glide_fep_preserves_existing_frozen_prime_candidates(project):
    original=database.fep_candidates(project.db_path,1,2,target_count=4)
    assert original[0]['ranking_source']=='prime'
    assert database.fep_candidates(project.db_path,1,2,target_count=4,docking_only=True)==original
