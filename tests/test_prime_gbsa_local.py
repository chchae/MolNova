"""Adam Prime batch ownership/recovery; mocks do not validate scientific scores."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from molnova import _core, database, md_gbsa, prime_gbsa_async as prime
from molnova import prime_gbsa_batch as batch, prime_gbsa_local as local
from molnova import prime_gbsa_local_native as calculator, schrodinger_guard as guard
from molnova.config import prime_gbsa_local_settings
from molnova.md_gbsa_native import save
from molnova.md_gbsa_local import digest

prepare_batch_inputs=local.prepare


@pytest.mark.parametrize('key,value', [('compute','true'),('cpus',17),('cpus',0),('cpus',True),
    ('cpus',1.5),('host','-adam'),('host','adam:16'),('work-dir','relative'),('schrodinger','relative')])
def test_local_prime_config_is_strict(key,value):
    with pytest.raises(ValueError,match='prime-gbsa-local'):
        prime_gbsa_local_settings({'prime-gbsa-local-'+key:value})


@pytest.fixture
def project(tmp_path,monkeypatch):
    db=tmp_path/'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany("INSERT INTO compound(name,smiles,iteration,state,docking_score) "
                         "VALUES (?,'C',1,'prime_gbsa_running',-10)",[('one',),('two',)])
    args=SimpleNamespace(db_path=db,output=tmp_path,schrodinger=Path('/suite'),host='old:7',
        prime_gbsa_flexdist=0.,prime_gbsa_local_compute=True,prime_gbsa_elite_count=1,
        **{k:v for k,v in prime_gbsa_local_settings({'prime-gbsa-local-work-dir':str(tmp_path/'share')}).items()
           if k!='prime_gbsa_local_compute'})
    directory=tmp_path/'iter01/08-prime_gbsa'
    directory.mkdir(parents=True)
    (directory/'prime_gbsa_input.maegz').write_bytes(b'input PV')
    monkeypatch.setattr(guard,'active_jobs',lambda *a:[])
    record=local.create_batch(args,directory,[1,2])
    root=Path(record['backend']['root'])
    (root/'prime').mkdir(parents=True)
    (root/'input').mkdir()
    save(root/'owner.json',record['owner'])
    (root/'input/prime_gbsa.maegz').write_bytes(b'input PV')
    (root/'prime/prime_gbsa-out.maegz').write_bytes(b'optimized output')
    save(root/'prime/prime.job.json',dict(command=record['command'],job_id='adam-0-1234abcd'))
    state=dict(status='done',native=dict(job_id='adam-0-1234abcd',launching=False),
        jobs=[dict(jobId='adam-0-1234abcd',status='DONE')],scores={'1':-50.,'2':-60.},missing_ids=[],
        output_sha256=digest(root/'prime/prime_gbsa-out.maegz'),diagnostics='')
    monkeypatch.setattr(local,'prepare',lambda *a:None)
    monkeypatch.setattr(guard,'register_md_remote',lambda *a:None)
    monkeypatch.setattr(guard,'unregister_md_remote',lambda *a:None)
    monkeypatch.setattr(guard,'submission_guard',lambda *a,**kw:nullcontext())
    def remote(config,action,where,*extra):
        if action=='verify':
            assert database.prime_gbsa_scored_ids(db,1)==[1,2]  # Verify/cleanup AFTER commit.
        return 'MOLNOVA_MD='+json.dumps(state)
    monkeypatch.setattr(md_gbsa,'remote_call',remote)
    def finish(iteration,args_,directory_,ids):
        scores=directory_/'scores.tsv'
        scores.write_text('id\tprime_gbsa_score\n1\t-50\n2\t-60\n')
        return _core.update_prime_gbsa_scores(scores,args_.db_path,allowed_ids=ids)
    monkeypatch.setattr(_core,'finish_iteration_prime_gbsa',finish)
    monkeypatch.setattr(batch,'publish_coordinates',lambda *a:(directory/'prime_gbsa_final.maegz').write_bytes(b'final complexes'))
    return args,directory,record,root,state


def test_local_prime_uses_one_sixteen_slot_batch_and_pins_defaults(project):
    args,directory,record,root,state=project
    cmd=record['command']
    assert cmd[cmd.index('-HOST')+1]=='localhost:16'
    assert cmd[cmd.index('-NJOBS')+1]=='2'
    assert '-rflexdist' not in cmd and '-frozen' not in cmd
    assert record['owner']['prime_gbsa_flexdist']==0
    args.prime_gbsa_local_compute=False
    assert record['backend']['ssh_host']=='adam'


def test_local_prime_flexibility_keeps_sidechains_and_configured_distance(project,monkeypatch):
    args,directory,record,root,state=project
    prime.job_file(directory).unlink()
    args.prime_gbsa_flexdist=6
    monkeypatch.setattr(batch,'prepare_complexes',lambda *a:(directory/'prime_gbsa.maegz').write_bytes(b'complexes'))
    saved=local.create_batch(args,directory,[1,2])
    assert saved['command'][-8:]==['-ligand','atom.i_molnova_batch_ligand 1',
        '-rflexdist','6.0','-rflexgroup','side','-out_type','COMPLEX']
    assert saved['owner']['input_sha256']==digest(directory/'prime_gbsa.maegz')


def test_saved_local_mode_recovers_even_when_flag_is_disabled(project,monkeypatch):
    args,directory,record,root,state=project
    args.prime_gbsa_local_compute=False
    monkeypatch.setattr(local,'poll_batch',lambda *a:'recovered exact Adam batch')
    assert _core.run_iteration_prime_gbsa(1,args,directory.parent,Path('/glide'))=='recovered exact Adam batch'


def test_enabling_local_mode_does_not_move_existing_native_batch(project,monkeypatch):
    args,directory,record,root,state=project
    record['mode']='native_batch_v1'
    save(prime.job_file(directory),record)
    monkeypatch.setattr(batch,'poll_batch',lambda *a:'existing native command retained')
    monkeypatch.setattr(local,'poll_batch',lambda *a:pytest.fail('migrated submitted job'))
    assert _core.run_iteration_prime_gbsa(1,args,directory.parent,Path('/glide'))=='existing native command retained'


def test_success_collects_scores_and_coordinates_before_shared_cleanup(project):
    args,directory,record,root,state=project
    assert local.poll_batch(1,args,directory,record)==2
    assert not root.exists()
    assert prime.load_job(directory) is None
    assert database.prime_gbsa_scored_ids(args.db_path,1)==[1,2]
    assert (directory/'prime_gbsa-out.maegz').read_bytes()==b'optimized output'
    assert (directory/'prime_gbsa_final.maegz').exists()
    assert list(directory.glob('adam-batch-*.completed.json'))


def test_database_failure_preserves_shared_inputs_and_native_identity(project,monkeypatch):
    args,directory,record,root,state=project
    def interrupted(*a):raise RuntimeError('database commit interrupted')
    monkeypatch.setattr(_core,'finish_iteration_prime_gbsa',interrupted)
    with pytest.raises(RuntimeError,match='commit interrupted'):
        local.poll_batch(1,args,directory,record)
    assert root.exists()
    assert prime.load_job(directory)['job_id']=='adam-0-1234abcd'
    assert database.prime_gbsa_scored_ids(args.db_path,1)==[]


def test_cleanup_restart_does_not_relaunch_or_reimport(project,monkeypatch):
    args,directory,record,root,state=project
    remove=local.shutil.rmtree
    def interrupted(path,*a,**kw):raise RuntimeError('cleanup interrupted')
    monkeypatch.setattr(local.shutil,'rmtree',interrupted)
    with pytest.raises(RuntimeError,match='cleanup interrupted'):
        local.poll_batch(1,args,directory,record)
    assert database.prime_gbsa_scored_ids(args.db_path,1)==[1,2]
    assert not root.exists() and root.with_name(root.name+'.collected').exists()
    monkeypatch.setattr(local.shutil,'rmtree',remove)
    monkeypatch.setattr(local,'prepare',lambda *a:pytest.fail('relaunch during cleanup'))
    assert local.poll_batch(1,args,directory,prime.load_job(directory))==0
    assert not root.with_name(root.name+'.collected').exists()


def test_unknown_missing_output_keeps_batch_for_reconciliation(project):
    args,directory,record,root,state=project
    state['missing_ids']=[2]
    with pytest.raises(prime.PrimeReconciliationPending,match='without confirmed'):
        local.poll_batch(1,args,directory,record)
    assert root.exists() and prime.load_job(directory)
    assert database.prime_gbsa_scored_ids(args.db_path,1)==[]


def test_successful_final_scores_override_transient_atomtyping_diagnostics(project):
    args,directory,record,root,state=project
    (root/'prime/prime_gbsa.Prime.log').write_text('Prime-MMGBSA: CMPID_1\ncannot atom type structure\n')
    assert local.poll_batch(1,args,directory,record)==2
    assert database.prime_gbsa_scored_ids(args.db_path,1)==[1,2]
    assert not root.exists()


def test_license_retry_is_durable_and_does_not_reset_unknown_jobs(project):
    args,directory,record,root,state=project
    state.update(status='failed',diagnostics='license checkout failed')
    assert local.poll_batch(1,args,directory,record) is None
    saved=prime.load_job(directory)
    assert saved['attempt']==1 and saved['recovery_token'] and saved['retry_at']>0
    assert root.exists()
    assert local.poll_batch(1,args,directory,saved) is None
    assert prime.load_job(directory)['attempt']==1


@pytest.mark.parametrize('entries,valid', [
    ([('CMPID_1',-10.),('CMPID_2',-20.)],True),
    ([('CMPID_1',-10.),('CMPID_1',-20.)],False),
    ([('CMPID_999',-10.)],False),([('CMPID_1',float('nan'))],False)])
def test_native_prime_output_validation_keeps_original_compound_identity(monkeypatch,entries,valid):
    structures=[SimpleNamespace(title=title,property={'r_psp_MMGBSA_dG_Bind':score}) for title,score in entries]
    monkeypatch.setitem(sys.modules,'schrodinger',SimpleNamespace(
        structure=SimpleNamespace(StructureReader=lambda *a:iter(structures))))
    if valid:assert calculator.read_scores(Path('/output'),dict(compound_ids=[1,2]))=={1:-10.,2:-20.}
    else:
        with pytest.raises(ValueError):calculator.read_scores(Path('/output'),dict(compound_ids=[1,2]))


def test_native_prime_defers_while_md_has_uncertain_reservation(project,monkeypatch):
    args,directory,record,root,state=project
    (root/'prime/prime.job.json').unlink()
    (root/'prime/prime_gbsa-out.maegz').unlink()
    peer=root.parent/'molnova-md/CMPID_7/replica01/prime'
    peer.mkdir(parents=True)
    save(peer/'prime.job.json',dict(job_id=None))
    monkeypatch.setattr(calculator.native,'poll_job',lambda *a:pytest.fail('MD reservation exceeded'))
    assert calculator.poll(root,Path('/suite'))['status']=='pending'


def test_streamed_scores_survive_parent_failure_without_duplicate_import(project,monkeypatch):
    args,directory,record,root,state=project
    child='adam-0-1234abce'
    path=root/'subjobs'/child/'result.maegz'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'completed child')
    state.update(status='running',subjobs=[dict(job_id=child,sha256=digest(path))])
    def extract(command,**kw):Path(command[-1]).write_text('id\tprime_gbsa_score\n1\t-50\n')
    monkeypatch.setattr(_core,'run_command',extract)
    assert local.poll_batch(1,args,directory,record) is None
    assert database.prime_gbsa_scored_ids(args.db_path,1)==[1]
    saved=prime.load_job(directory)
    assert saved['imported_subjobs']==[child]
    state.update(status='failed',subjobs=[])
    with pytest.raises(prime.PrimeReconciliationPending):local.poll_batch(1,args,directory,saved)
    assert database.prime_gbsa_scored_ids(args.db_path,1)==[1]
    assert root.exists()


def test_missing_shared_folder_cannot_recreate_a_saved_submission(project,monkeypatch):
    args,directory,record,root,state=project
    import shutil
    shutil.rmtree(root)
    record['prepared']=True
    # The real preparation function is tested, without invoking any SSH.
    monkeypatch.setattr(md_gbsa,'remote_call',lambda *a:pytest.fail('resubmitted missing shared work'))
    with pytest.raises(RuntimeError,match='no recreation/resubmission'):
        prepare_batch_inputs(directory,record)
    assert not root.exists()


def test_native_license_recovery_never_moves_live_or_successful_job(project,monkeypatch):
    args,directory,record,root,state=project
    monkeypatch.setattr(calculator.native,'native_rows',lambda *a:[dict(jobId='adam-0-1234abcd',status='RUNNING')])
    with pytest.raises(RuntimeError,match='live/uncertain/successful'):
        calculator.retry(root,Path('/suite'),'a'*32)
    assert (root/'prime/prime.job.json').exists()
    assert not (root/'license-retries').exists()


def test_native_license_recovery_archives_and_is_idempotent(project,monkeypatch):
    args,directory,record,root,state=project
    monkeypatch.setattr(calculator.native,'native_rows',lambda *a:[dict(jobId='adam-0-1234abcd',status='FAILED')])
    save(root/'launch-intent.json',dict(started_at=1))
    assert calculator.retry(root,Path('/suite'),'a'*32)['status']=='ready'
    assert not (root/'prime').exists()
    assert not (root/'launch-intent.json').exists()
    assert (root/'license-retries'/('a'*32)/'prime/prime.job.json').exists()
    assert calculator.retry(root,Path('/suite'),'a'*32)['status']=='ready'


def test_unjournaled_native_output_is_not_resubmitted(project,monkeypatch):
    args,directory,record,root,state=project
    (root/'prime/prime.job.json').unlink()
    monkeypatch.setattr(calculator.native,'poll_job',lambda *a:pytest.fail('duplicate calculation'))
    with pytest.raises(RuntimeError,match='Unjournaled'):
        calculator.poll(root,Path('/suite'))


@pytest.mark.parametrize('status,downloaded,expected',[('RUNNING',True,0),('DONE',False,0),('DONE',True,1)])
def test_only_verified_completed_legacy_child_outputs_are_streamed(project,monkeypatch,status,downloaded,expected):
    args,directory,record,root,state=project
    child='adam-0-1234abce';parent='adam-0-1234abcd'
    launch=directory/'native-child-launch'
    launch.mkdir()
    (launch/'prime_gbsa.00001-out.maegz').write_bytes(b'completed immutable output')
    job=SimpleNamespace(isComplete=lambda:True,ExitStatus='finished',isDownloaded=lambda:downloaded,Dir=str(launch))
    monkeypatch.setitem(sys.modules,'schrodinger',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'schrodinger.job',SimpleNamespace(jobcontrol=SimpleNamespace(Job=lambda *a:job)))
    monkeypatch.setattr(calculator,'read_scores',lambda *a:{1:-50.})
    state=dict(native=dict(job_id=parent),jobs=[dict(jobId=child,parentJobId=parent,status=status,jobName='prime_gbsa.00001')])
    outputs=calculator.subjobs(root,record['owner'],state,Path('/suite'))
    assert len(outputs)==expected
    assert (root/'subjobs'/child/'result.maegz').exists()==bool(expected)
