"""Adam transport/recovery regressions; mocked energies do not validate science."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, md_gbsa, md_gbsa_local as local
from molnova import md_gbsa_local_native as calculator, md_gbsa_native as native
from molnova import schrodinger_guard as guard
from molnova.config import md_gbsa_settings


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())


@pytest.fixture
def task(tmp_path):
    root = tmp_path/'share/molnova-run/CMPID_1/replica01'
    files = {'prime/frames.maegz': b'frames', 'frames.json': [dict(frame=0,time_ps=1000,ligand_rmsd=.2)],
             'identity.json': dict(id=1), 'md/md.job.json': dict(job_id='upstream',downloaded=True)}
    for name, value in files.items():
        write(root/name, value)
    config = dict(mode='adam_prime_v1',id=1,slots=16,source_root='/md/run/CMPID_1/replica01',
                  protocol=dict(snapshots=1,seed=1,prime_job_type='REAL_MIN',prime_flexdist=6.0,
                                prime_flexgroup='residue'),
                  files={name:local.digest(root/name) for name in files})
    write(root/'input.json',config)
    return root, config


@pytest.mark.parametrize('key,value', [('local-compute',1),('local-compute','true'),
    ('local-cpus',17),('local-cpus',0),('local-cpus',True),('local-cpus',2.5),
    ('local-host','-adam'),('local-work-dir','relative'),('local-schrodinger','relative')])
def test_config_rejects_invalid_local_settings(key,value):
    with pytest.raises(ValueError, match='md-gbsa'):
        md_gbsa_settings({'md-gbsa-'+key:value})


def test_local_defaults_and_legacy_protocol_stays_separate():
    config = md_gbsa_settings({})
    assert config['md_gbsa_local_compute'] is False
    assert config['md_gbsa_local_cpus'] == 16
    assert config['md_gbsa_local_host'] == 'adam'
    assert config['md_gbsa_local_work_dir'] == '/home1/prime_share'
    settings = dict(root='/remote/run',protocol=dict(seed=1))
    assert local.add_settings(SimpleNamespace(),settings) == settings
    assert 'prime_job_type' not in settings['protocol']


def test_adam_command_is_local_minimization_with_sixteen_slots(task,monkeypatch):
    root, config = task
    commands = []
    monkeypatch.setattr(native,'active_jobs',lambda *a:[])
    monkeypatch.setattr(native,'poll_job',lambda folder,phase,cmd,suite:
                        commands.append(cmd) or dict(status='running'))
    assert calculator.poll(root,Path('/adam/suite'))['status'] == 'running'
    cmd = commands[0]
    assert cmd[cmd.index('-HOST')+1] == 'localhost:16'
    assert cmd[cmd.index('-NJOBS')+1] == '1'
    assert cmd[cmd.index('-job_type')+1] == 'REAL_MIN'
    assert cmd[cmd.index('-rflexdist')+1] == '6.0'
    assert cmd[cmd.index('-rflexgroup')+1] == 'residue'
    assert '-frozen' not in cmd and not any('slurm' in part for part in cmd)


def test_changed_frames_cannot_launch(task,monkeypatch):
    root,_ = task
    (root/'prime/frames.maegz').write_bytes(b'changed')
    monkeypatch.setattr(native,'poll_job',lambda *a:pytest.fail('invalid input submitted'))
    with pytest.raises(RuntimeError,match='Frozen Adam input changed'):
        calculator.poll(root,Path('/suite'))


@pytest.mark.parametrize('status',['glide','prime_mmgbsa'])
def test_existing_adam_job_prevents_another_parent(task,monkeypatch,status):
    root,_ = task
    monkeypatch.setattr(native,'active_jobs',lambda *a:[dict(jobName=status)])
    monkeypatch.setattr(native,'poll_job',lambda *a:pytest.fail('conflicting submission'))
    assert calculator.poll(root,Path('/suite'))['status'] == 'pending'


def test_uncertain_other_project_reserves_adam_without_registered_native_job(task,monkeypatch):
    root,_ = task
    write(root.parents[2]/'molnova-other/CMPID_8/replica01/prime/prime.job.json',dict(job_id=None,launching=True))
    monkeypatch.setattr(native,'active_jobs',lambda *a:[])
    monkeypatch.setattr(native,'poll_job',lambda *a:pytest.fail('uncertain launch ignored'))
    assert calculator.poll(root,Path('/suite'))['reason'] == 'Uncertain Adam launch'


def test_saved_active_parent_reserves_slots_even_when_absent_from_active_list(task,monkeypatch):
    root,_ = task
    write(root.parents[2]/'molnova-other/CMPID_8/replica01/prime/prime.job.json',dict(job_id='saved'))
    monkeypatch.setattr(native,'active_jobs',lambda *a:[])
    monkeypatch.setattr(native,'native_rows',lambda *a:[dict(jobId='saved',status='RUNNING')])
    monkeypatch.setattr(native,'poll_job',lambda *a:pytest.fail('sixteen-slot reservation ignored'))
    assert calculator.poll(root,Path('/suite'))['status'] == 'pending'


def test_download_validates_before_publish_and_does_not_overwrite(task,monkeypatch,tmp_path):
    def transfer(command,stdout,**kw):
        stdout.write(b'frames')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(local.subprocess,'run',transfer)
    target = tmp_path/'transfer/frames.maegz'
    with pytest.raises(RuntimeError,match='checksum'):
        local.download('slurm','/source',target,'wrong')
    assert not target.exists()
    local.download('slurm','/source',target,hashlib.sha256(b'frames').hexdigest())
    target.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Frozen shared frame'):
        local.download('slurm','/source',target,hashlib.sha256(b'frames').hexdigest())
    assert target.read_bytes() == b'changed'


@pytest.mark.parametrize('status',['RUNNING','FAILED'])
def test_verification_refuses_live_or_failed_native_parent(task,monkeypatch,status):
    root,config = task
    write(root/'prime/prime.job.json',dict(job_id='prime'))
    monkeypatch.setattr(native,'native_rows',lambda *a:[dict(jobId='prime',status=status)])
    with pytest.raises(RuntimeError,match='active, failed or uncertain'):
        calculator.verify(root,Path('/suite'))


def test_retry_archives_only_prime_and_resumes_interrupted_moves(task,monkeypatch):
    root,config = task
    token = 'a'*32
    write(root/'prime/prime.job.json',dict(job_id='failed'))
    monkeypatch.setattr(native,'native_rows',lambda *a:[dict(jobId='failed',status='FAILED')])
    assert calculator.retry(root,Path('/suite'),token)['status'] == 'ready'
    assert (root/'md/md.job.json').exists()
    assert (root/'retry_history'/token/'prime/prime.job.json').exists()
    assert not (root/'prime/prime.job.json').exists()
    # Simulate a stop after the directory move, before copying frames back.
    (root/'prime/frames.maegz').unlink()
    assert calculator.retry(root,Path('/suite'),token)['status'] == 'ready'
    assert local.digest(root/'prime/frames.maegz') == config['files']['prime/frames.maegz']


@pytest.mark.parametrize('job_id,status',[(None,'DONE'),('job','RUNNING')])
def test_retry_never_rearms_uncertain_or_live_work(task,monkeypatch,job_id,status):
    root,_ = task
    write(root/'prime/prime.job.json',dict(job_id=job_id))
    monkeypatch.setattr(native,'native_rows',lambda *a:[dict(jobId='job',status=status)])
    with pytest.raises(RuntimeError,match='JobId|active or uncertain'):
        calculator.retry(root,Path('/suite'),'b'*32)
    assert not (root/'retry_history').exists()


@pytest.fixture
def project(task,tmp_path,monkeypatch):
    root,config = task
    db = tmp_path/'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,docking_score,prime_gbsa_score) "
                     "VALUES ('one','C',1,'prime_gbsa_done',-10,-50)")
    settings = dict(root='/md/run',protocol=config['protocol'],replicates=1,ssh_host='slurm',
        suite='/md/suite',helper='/md/native',gpus=1,
        local_compute=dict(mode='adam_prime_v1',root=str(root.parents[1]),ssh_host='adam',
                           suite='/adam/suite',helper='/adam/helper',slots=16))
    write(root.parents[1]/'owner.json',local.owner(settings))
    database.freeze_md_gbsa_selection(db,1,settings,count=1,target_count=1,prime_count=1)
    args = SimpleNamespace(db_path=db,output=tmp_path,schrodinger=Path('/suite'),md_gbsa_enabled=True)
    directory = tmp_path/'iter01/09-md_gbsa'
    data = dict(status='done',id=1,mean=-70.,std=0.,protocol=config['protocol'],
                snapshots=[dict(frame=0,score=-70.,ligand_rmsd=.2)])
    archive = directory/'adam-results/CMPID_1/replica01'
    write(archive/'prime/prime_mmgbsa-out.maegz',b'coordinates')
    write(archive/'collected.json',dict(result=data,output_sha256=local.digest(archive/'prime/prime_mmgbsa-out.maegz')))
    monkeypatch.setattr(md_gbsa,'remote_call',lambda *a,**kw:'MOLNOVA_MD='+json.dumps(data))
    return args,settings,directory,data,root


def test_cleanup_only_after_database_commit_and_keeps_other_runs(project):
    args,settings,directory,data,root = project
    sibling = root.parents[2]/'another-project'
    sibling.mkdir()
    local.cleanup_committed(args,1,settings,directory)
    assert root.exists()  # Even validated/collected native output cannot delete inputs yet.
    database.store_md_gbsa_result(args.db_path,1,1,data)
    assert 'cleanup pending' in database.md_gbsa_completion_reason(args.db_path,1)
    local.cleanup_committed(args,1,settings,directory)
    assert not root.parent.exists()
    assert sibling.exists()
    assert database.md_gbsa_completion_reason(args.db_path,1) is None
    local.cleanup_committed(args,1,settings,directory)  # Idempotent after deletion.
    local.finish_cleanup(settings,[1])
    assert not Path(settings['local_compute']['root']).exists()
    assert (directory/'adam-results/CMPID_1/replica01/prime/prime_mmgbsa-out.maegz').exists()


def test_cleanup_retains_files_on_collection_corruption_or_unknown_native_status(project,monkeypatch):
    args,settings,directory,data,root = project
    database.store_md_gbsa_result(args.db_path,1,1,data)
    def unknown(*a,**kw):
        raise RuntimeError('JobServer unavailable')
    monkeypatch.setattr(md_gbsa,'remote_call',unknown)
    with pytest.raises(RuntimeError,match='JobServer unavailable'):
        local.cleanup_committed(args,1,settings,directory)
    assert root.exists()
    assert database.md_gbsa_completion_reason(args.db_path,1)
    (directory/'adam-results/CMPID_1/replica01/prime/prime_mmgbsa-out.maegz').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError,match='coordinates changed'):
        local.cleanup_committed(args,1,settings,directory)
    assert root.exists()


def test_cleanup_restart_after_atomic_rename_needs_no_remote_job(project,monkeypatch):
    args,settings,directory,data,root = project
    database.store_md_gbsa_result(args.db_path,1,1,data)
    write(directory/'adam-results/CMPID_1/cleanup.json',dict(owner=local.owner(settings),id=1,results=[data]))
    root.parent.rename(root.parents[1]/'.collected-CMPID_1')
    monkeypatch.setattr(md_gbsa,'remote_call',lambda *a:pytest.fail('already verified before rename'))
    local.cleanup_committed(args,1,settings,directory)
    assert database.md_gbsa_completion_reason(args.db_path,1) is None
    assert not (root.parents[1]/'.collected-CMPID_1').exists()


def test_worker_restart_finishes_cleanup_without_recomputing(project,monkeypatch):
    args,settings,directory,data,root = project
    database.store_md_gbsa_result(args.db_path,1,1,data)
    monkeypatch.setattr(guard,'unregister_md_remote',lambda *a:None)
    monkeypatch.setattr(md_gbsa,'prepare',lambda *a:pytest.fail('recomputed committed MD'))
    md_gbsa.run(args,1)
    assert database.md_gbsa_completion_reason(args.db_path,1) is None
    assert not root.parents[1].exists()


def test_missing_shared_inputs_without_cleanup_receipt_stays_pending(project):
    import shutil
    args,settings,directory,data,root = project
    database.store_md_gbsa_result(args.db_path,1,1,data)
    shutil.rmtree(root.parents[1])
    with pytest.raises(RuntimeError,match='disappeared before verified cleanup'):
        local.cleanup_committed(args,1,settings,directory)
    assert database.pending_md_local_cleanup_ids(args.db_path,1) == {1}
    assert database.md_gbsa_completion_reason(args.db_path,1)


def test_registry_includes_adam_and_slurm_and_unregisters_both(tmp_path,monkeypatch):
    registry = tmp_path/'registry.json'
    monkeypatch.setattr(guard,'_md_registry',lambda:registry)
    settings = dict(ssh_host='slurm',suite='/md',root='/md/run',helper='/md/helper',
        local_compute=dict(ssh_host='adam',suite='/adam',root='/share/run',helper='/share/helper'))
    guard.register_md_remote(settings)
    assert {s['ssh_host'] for s in json.loads(registry.read_text()).values()} == {'slurm','adam'}
    guard.unregister_md_remote(settings)
    assert json.loads(registry.read_text()) == {}


@pytest.mark.parametrize('fail_commit',[False,True])
def test_worker_transfers_collects_commits_then_cleans_and_recovers_commit_failure(project,monkeypatch,fail_commit):
    args,settings,directory,data,root = project
    config = json.loads((root/'input.json').read_text())
    write(root/'prime/prime_mmgbsa-out.maegz',b'actual coordinates')
    write(root/'prime/prime.job.json',dict(job_id='adam-job'))
    write(root/'result.json',data)
    monkeypatch.setattr(md_gbsa,'prepare',lambda *a:None)
    monkeypatch.setattr(local,'prepare',lambda *a:None)
    monkeypatch.setattr(guard,'register_md_remote',lambda *a:None)
    monkeypatch.setattr(guard,'unregister_md_remote',lambda *a:None)
    monkeypatch.setattr(guard,'submission_guard',lambda *a,**kw:nullcontext())
    monkeypatch.setattr(guard,'active_jobs',lambda *a:[])
    def remote(config_,action,path,*extra):
        if action=='frames':
            return 'MOLNOVA_MD='+json.dumps(dict(status='frames',files=config['files']))
        assert action in ('poll','verify')
        return 'MOLNOVA_MD='+json.dumps(data)
    monkeypatch.setattr(md_gbsa,'remote_call',remote)
    original = database.store_md_gbsa_result
    if fail_commit:
        def failed(*a):
            raise RuntimeError('commit interrupted')
        monkeypatch.setattr(database,'store_md_gbsa_result',failed)
        with pytest.raises(RuntimeError,match='commit interrupted'):
            md_gbsa.run(args,1)
        assert root.exists()
        assert database.md_gbsa_rows(args.db_path,1)[1][1] is None
        monkeypatch.setattr(database,'store_md_gbsa_result',original)
    md_gbsa.run(args,1)
    assert database.md_gbsa_completion_reason(args.db_path,1) is None
    assert not root.parents[1].exists()
    assert (directory/'adam-results/CMPID_1/replica01/prime/prime_mmgbsa-out.maegz').read_bytes() == b'actual coordinates'


def test_cross_project_fence_allows_md_frame_preparation_but_no_adam_submission(task,monkeypatch):
    root,config = task
    settings = dict(local_compute=dict(root=str(root.parents[1])), protocol=config['protocol'])
    (root/'prime/frames.maegz').unlink()
    monkeypatch.setattr(md_gbsa,'remote_call',lambda *a:'MOLNOVA_MD='+json.dumps(dict(status='frames')))
    assert local.poll(settings,1,0,'/md',root/'archive',allow_submit=False)['status'] == 'pending'


def test_prepare_checks_same_shared_mount_and_preserves_frozen_helpers(tmp_path,monkeypatch):
    args = SimpleNamespace(md_gbsa_local_compute=True, md_gbsa_local_host='adam',
        md_gbsa_local_work_dir=str(tmp_path), md_gbsa_local_schrodinger='/suite',md_gbsa_local_cpus=16)
    settings = local.add_settings(args,dict(root='/md/run',protocol=dict(seed=1)))
    monkeypatch.setattr(md_gbsa,'remote_call',lambda *a:'MOLNOVA_MD='+json.dumps(dict(owner={'foreign':True})))
    with pytest.raises(RuntimeError,match='does not see'):
        local.prepare(settings)
    monkeypatch.setattr(md_gbsa,'remote_call',lambda *a:'MOLNOVA_MD='+json.dumps(dict(owner=local.owner(settings))))
    local.prepare(settings)
    helper = Path(settings['local_compute']['helper'])
    helper.write_text('modified')
    with pytest.raises(RuntimeError,match='helper changed'):
        local.prepare(settings)
