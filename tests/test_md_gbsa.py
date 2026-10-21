"""Workflow regressions; mocked energies do not validate scientific calculations."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from molnova import _core, database, driver, md_gbsa, md_gbsa_native as native
from molnova.config import md_gbsa_settings
from molnova.paths import stage_directory
from molnova.states import CompoundState as State


@pytest.fixture
def db(tmp_path):
    path=tmp_path/'project.sqlite'
    with database.connect(path) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany('INSERT INTO compound(name,smiles,iteration,state,docking_score,prime_gbsa_score) '
                         'VALUES (?,\'C\',1,?,?,?)',
                         [(f'cmp{i}',State.PRIME_GBSA_DONE,-10+i,-60+i) for i in range(4)])
    return path


def freeze(db, **kw):
    settings=dict(protocol=dict(snapshots=20),replicates=1,root='/immutable')
    return database.freeze_md_gbsa_selection(db,1,settings,target_count=4,prime_count=4,**kw)


def result(cid,score):
    return dict(id=cid,mean=score,std=0.,snapshots=[dict(frame=i,score=score) for i in range(20)])


def test_final_prime_selection_is_frozen_before_claim_and_no_refill(db):
    saved=freeze(db)
    assert saved['ids']==[1,2]
    with database.connect(db) as conn:
        assert conn.execute('SELECT state FROM compound WHERE id=1').fetchone()[0]==State.MD_GBSA_RUNNING
        conn.execute('UPDATE compound SET prime_gbsa_score=-200 WHERE id=4')
        conn.execute('UPDATE compound SET state=?,failed_stage=? WHERE id=1',(State.FAILED,'md_gbsa'))
    assert freeze(db,count=4)['ids']==[1,2]
    assert database.md_gbsa_completion_reason(db,1)
    database.store_md_gbsa_result(db,1,2,result(2,-70))
    assert database.md_gbsa_completion_reason(db,1) is None
    assert 'need 2' in database.md_gbsa_completion_reason(db,1,2)


@pytest.mark.parametrize('state', [State.GLIDE_RUNNING,State.PRIME_GBSA_RUNNING,State.AMBER_RUNNING])
def test_no_selection_while_upstream_runs(db,state):
    with database.connect(db) as conn:
        conn.execute('UPDATE compound SET state=? WHERE id=4',(state,))
    with pytest.raises(RuntimeError,match='waiting'):
        freeze(db)
    assert database.md_gbsa_selection(db,1) is None


def test_complete_ensemble_drives_elites_and_fep_without_overwriting_prime(db):
    freeze(db)
    database.store_md_gbsa_result(db,1,1,result(1,-50))
    database.store_md_gbsa_result(db,1,2,result(2,-80))
    assert [r[0] for r in database.iteration_md_gbsa_elites(db,1,2)]==[2,1]
    fep=database.fep_candidates(db,1,2,target_count=4,prime_count=4)
    assert [r['id'] for r in fep]==[2,1]
    assert all(r['ranking_source']=='md' for r in fep)
    assert fep[0]['prime_gbsa_score']==-59
    assert fep[0]['md_gbsa_score']==-80
    with database.connect(db) as conn:
        assert conn.execute('SELECT md_gbsa_snapshots,md_gbsa_std FROM compound WHERE id=2').fetchone()==(20,0)


def test_no_partial_ensemble_or_prime_fallback(db):
    freeze(db)
    with pytest.raises(RuntimeError,match='waiting'):
        database.fep_candidates(db,1,1,target_count=4,prime_count=4)
    partial=result(1,-50);partial['snapshots'].pop()
    with pytest.raises(ValueError,match='Incomplete'):
        database.store_md_gbsa_result(db,1,1,partial)
    assert database.md_gbsa_rows(db,1)[1][1] is None


def test_conflicting_results_rejected_but_identical_retry_is_safe(db):
    freeze(db)
    database.store_md_gbsa_result(db,1,1,result(1,-50))
    database.store_md_gbsa_result(db,1,1,result(1,-50))
    with pytest.raises(ValueError,match='Conflicting'):
        database.store_md_gbsa_result(db,1,1,result(1,-51))


def test_frozen_fep_is_not_silently_replaced_by_md(db):
    old=database.fep_candidates(db,1,2,target_count=4,prime_count=4)
    freeze(db)
    for cid in (1,2):
        database.store_md_gbsa_result(db,1,cid,result(cid,-50))
    with pytest.raises(RuntimeError,match='predates MD'):
        database.fep_candidates(db,1,2,target_count=4,prime_count=4)
    assert database.fep_candidates(db,1,2,for_import=True)[0]['id']==old[0]['id']


def test_enabled_driver_backfills_only_latest_and_recovers_when_disabled(db):
    args=SimpleNamespace(db_path=db,output=db.parent,target_count=4,prime_gbsa_input_count=4,
                         max_iteration=3,md_gbsa_enabled=True)
    assert driver.choose_iteration(args)==1
    freeze(db);args.md_gbsa_enabled=False
    assert driver.choose_iteration(args)==1
    for cid in (1,2):
        database.store_md_gbsa_result(db,1,cid,result(cid,-50))
    assert driver.choose_iteration(args)==2
    assert [s[0] for s in driver.build_stages(False,True,False,True)][-3:]==['prime_gbsa','md-gbsa','fep']


def test_flag_disabled_leaves_no_remote_work(db,monkeypatch):
    monkeypatch.setattr(md_gbsa,'ssh',lambda *a,**kw:pytest.fail('unexpected SSH'))
    md_gbsa.run(SimpleNamespace(db_path=db,md_gbsa_enabled=False),1)


@pytest.mark.parametrize('key,value', [
    ('snapshots',19),('snapshots',101),('production-ns',0),('production-ns',float('nan')),
    ('equilibration-ns',-1),('replicates',False),('gpus',21),('cpus',385),('input-ratio',1.1),
    ('input-count',0),('analysis-start-fraction',1),('gpu-host','-bad'),('schrodinger','relative'),
])
def test_protocol_validation(key,value):
    with pytest.raises(ValueError,match='md-gbsa'):
        md_gbsa_settings({'md-gbsa-'+key:value})


def test_unknown_native_launch_never_resubmits(tmp_path,monkeypatch):
    command=['native','input']
    native.save(tmp_path/'md.job.json',dict(command=command,job_id=None,launching=True))
    monkeypatch.setattr(native.subprocess,'Popen',lambda *a,**kw:pytest.fail('duplicate submission'))
    monkeypatch.setattr(native.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout='',stderr='No jobs matching your search criteria were found.'))
    assert native.poll_job(tmp_path,'md',command,Path('/suite'))['status']=='pending'


def test_parent_done_is_insufficient_with_live_child(tmp_path,monkeypatch):
    native.save(tmp_path/'md.job.json',dict(command=['native'],job_id='saved'))
    monkeypatch.setattr(native,'native_rows',lambda *a:[dict(jobId='saved',status='DONE'),dict(jobId='child',status='RUNNING')])
    monkeypatch.setattr(native.subprocess,'run',lambda *a,**kw:pytest.fail('download of live job'))
    assert native.poll_job(tmp_path,'md',['native'],Path('/suite'))['status']=='running'


def test_zero_exit_and_cached_output_do_not_prove_completion(tmp_path,monkeypatch):
    native.save(tmp_path/'md.job.json',dict(command=['native'],job_id='saved',client_exit=0))
    (tmp_path/'md-out.cms').touch()
    monkeypatch.setattr(native,'native_rows',lambda *a:[dict(jobId='saved',status='FAILED')])
    assert native.poll_job(tmp_path,'md',['native'],Path('/suite'))['status']=='failed'


def test_preparation_refuses_changed_source_before_transfer(tmp_path,monkeypatch):
    settings=dict(root='/remote',ssh_host='slurm',source=str(tmp_path/'source'),source_sha256='wrong',helper_sha256='wrong')
    (tmp_path/'source').write_text('modified')
    monkeypatch.setattr(md_gbsa,'ssh',lambda *a,**kw:'')
    with pytest.raises(RuntimeError,match='Frozen MD input changed'):
        md_gbsa.prepare(settings,[1])


def test_md_output_keeps_legacy_fep_but_new_fep_follows_md(tmp_path):
    assert stage_directory(tmp_path,1,'fep').name=='09-fep'
    stage_directory(tmp_path,1,'md_gbsa').mkdir(parents=True)
    assert stage_directory(tmp_path,1,'fep').name=='10-fep'
    (tmp_path/'iter01/09-fep').mkdir()
    assert stage_directory(tmp_path,1,'fep').name=='09-fep'
    (tmp_path/'iter01/10-fep').mkdir()
    with pytest.raises(RuntimeError,match='Ambiguous'):
        stage_directory(tmp_path,1,'fep')


def test_remote_prime_fences_local_prime_and_glide_but_allows_md_recovery(monkeypatch):
    from molnova import schrodinger_guard as guard
    remote=dict(jobId='remote-prime',jobName='prime_mmgbsa',molnova_md_remote=True)
    monkeypatch.setattr(guard,'active_jobs',lambda suite:[remote])
    for stage in ('glide','prime_gbsa'):
        with pytest.raises(guard.SubmissionBlocked,match='prime_gbsa still active'):
            with guard.submission_guard('/suite',stage):
                pytest.fail('conflicting launch admitted')
    with guard.submission_guard('/suite','prime_gbsa',md_controller=True):
        pass


@pytest.mark.parametrize('client,ok', [('1.19.4',True),('1.21.2',False)])
def test_jobserver_version_preflight(client,ok,monkeypatch):
    def execute(command,**kw):
        if 'hosts' in command:
            return SimpleNamespace(stdout='  server:8030\n      Name: slurm-gpu\n      Name: slurm-cpu\n')
        return SimpleNamespace(stdout=f'Server API version: v1.19.4\nClient API version: v{client}\n')
    monkeypatch.setattr(native.subprocess,'run',execute)
    if ok:
        assert native.preflight(Path('/suite'),'slurm-gpu','slurm-cpu')['status']=='ready'
    else:
        with pytest.raises(RuntimeError,match='Incompatible JobServer'):
            native.preflight(Path('/suite'),'slurm-gpu','slurm-cpu')


def test_duplicate_frame_identity_cannot_publish_mean(db):
    freeze(db)
    invalid=result(1,-50)
    invalid['snapshots'][1]['frame']=0
    with pytest.raises(ValueError,match='Duplicate'):
        database.store_md_gbsa_result(db,1,1,invalid)


def test_md_export_preserves_score_source_label(tmp_path,monkeypatch):
    monkeypatch.setattr(_core,'decompose_elite_for_libinvent',lambda *a:('C[*]','C*'))
    _core.write_elite_libinvent_training_files([(1,'x','CC',1,-10,-80)],'C[*]',tmp_path,score_source='md_gbsa')
    header=(tmp_path/'elite_selected.tsv').read_text().splitlines()[0]
    assert 'md_gbsa_score' in header and 'prime_gbsa_score' not in header


def test_concurrent_selection_has_one_frozen_owner(db):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        saved=list(pool.map(lambda n:freeze(db,count=n),[1,2,3,4]))
    assert all(s==saved[0] for s in saved)
    assert all(database.md_gbsa_rows(db,1)[cid][2]==State.MD_GBSA_RUNNING for cid in saved[0]['ids'])


@pytest.mark.parametrize('wrong_identity',[False,True])
def test_controller_recovers_disabled_flag_and_publishes_complete_ensembles(db,monkeypatch,wrong_identity):
    from contextlib import nullcontext
    from molnova import schrodinger_guard as guard
    settings=dict(protocol=dict(snapshots=20,seed=1),replicates=1,root='/frozen',
                  ssh_host='slurm',suite='/suite',helper='/helper',gpus=2)
    database.freeze_md_gbsa_selection(db,1,settings,target_count=4,prime_count=4)
    monkeypatch.setattr(md_gbsa,'prepare',lambda config,ids:None)
    monkeypatch.setattr(guard,'register_md_remote',lambda s:None)
    monkeypatch.setattr(guard,'unregister_md_remote',lambda s:None)
    monkeypatch.setattr(guard,'submission_guard',lambda *a,**kw:nullcontext())
    monkeypatch.setattr(guard,'active_jobs',lambda *a:[])
    def poll(config,action,root):
        cid=int(root.split('/CMPID_')[1].split('/')[0])
        data=result(cid, -50.0 if cid==1 else -80.0)
        data.update(status='done',protocol=settings['protocol'])
        if wrong_identity:
            data['id']=999
        for frame in data['snapshots']:
            frame['ligand_rmsd']=.5
        return 'MOLNOVA_MD='+json.dumps(data)
    monkeypatch.setattr(md_gbsa,'remote_call',poll)
    args=SimpleNamespace(db_path=db,output=db.parent,schrodinger=Path('/suite'),md_gbsa_enabled=False)
    if wrong_identity:
        with pytest.raises(RuntimeError,match='identity/protocol'):
            md_gbsa.run(args,1)
        assert database.md_gbsa_rows(db,1)[1][1] is None
    else:
        md_gbsa.run(args,1)
        assert database.md_gbsa_completion_reason(db,1) is None
        assert database.iteration_md_gbsa_elites(db,1,2)[0][0]==2
        assert (db.parent/'iter01/09-md_gbsa/CMPID_2.json').is_file()


def test_deleted_compounds_do_not_release_durable_remote_md_work(db,tmp_path):
    freeze(db)
    with database.connect(db) as conn:
        conn.execute('DELETE FROM compound')
    assert database.iteration_was_deleted(db,1)
    with pytest.raises(RuntimeError,match='durable MD work'):
        database.reset_deleted_iteration(db,1,tmp_path/'backup.sqlite')
    assert database.md_gbsa_selection(db,1)['ids']==[1,2]
