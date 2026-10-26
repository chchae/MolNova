"""Slurm lifecycle and pure-pool tests; native scheduling is tested separately."""
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from molnova.config import combichem_3d_settings
from molnova.stages import combichem_slurm as slurm
from molnova.stages import combichem_slurm_control as control


@pytest.fixture
def project(tmp_path):
    script=Path(__file__).resolve().parents[1]/'examples/combichem-3d/make_toy_inputs.py'
    spec=importlib.util.spec_from_file_location('slurm_toy',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    filename=module.create(tmp_path/'toy',0,2)
    extra='''execution-backend = "slurm"
remote-env-script = "/env/amber.sh"
worker-hosts = { cpu1 = 1, cpu2 = 1 }
slurm-time-limit = 300
slurm-memory-mb = 2048
'''
    filename.write_text(filename.read_text().replace('[filters]',extra+'[filters]'))
    return filename,combichem_3d_settings(filename)


def flags():
    return SimpleNamespace(export_only=False,debug=False,trace=False)


def test_slurm_configuration_and_generated_output_do_not_change_submission_identity(project):
    filename,cfg=project
    assert cfg['processes']==2 and cfg['slurm-partition']=='cpu'
    before=slurm.signature(cfg,[])
    generated=Path(cfg['reference-smiles'])
    cfg['reference-smiles']=str(Path(cfg['out-dir'])/'input-cache/references.smi')
    before=slurm.signature(cfg,[])
    target=Path(cfg['reference-smiles']);target.parent.mkdir(parents=True);target.write_text('CC\n')
    assert slurm.signature(cfg,[])==before
    filename.write_text(filename.read_text()+'\n# Changed queued input\n')
    assert slurm.signature(cfg,[])!=before


@pytest.mark.parametrize('line',[
    'slurm-time-limit = 0','slurm-memory-mb = -1','slurm-poll-seconds = 0',
    'slurm-poll-seconds = true','slurm-host = "-bad"','slurm-partition = "cpu;exit"',
    'worker-hosts = { cpu1 = 1, cpu2 = 2 }','execution-host = "cpu2"'])
def test_invalid_slurm_resources_are_rejected(project,line):
    filename,_=project
    key=line.split('=')[0].strip()
    text='\n'.join(row for row in filename.read_text().splitlines() if not row.startswith(key+' ='))
    filename.write_text(text.replace('[filters]',line+'\n[filters]'))
    with pytest.raises(ValueError):combichem_3d_settings(filename)


def test_submission_is_native_and_idempotent(project,monkeypatch,tmp_path):
    _,cfg=project
    root=tmp_path/'native'
    slurm.create_request(cfg,[],root,'digest')
    submit=Mock(return_value='123;cluster\n')
    monkeypatch.setattr(control.subprocess,'check_output',submit)
    assert control.submit(root)['job_id']=='123'
    assert control.submit(root)['job_id']=='123'
    assert submit.call_count==1
    command=submit.call_args.args[0]
    assert '--hint=nomultithread' in command and '--ntasks-per-node=1' in command
    assert command[command.index('--nodelist')+1]=='cpu1,cpu2'
    assert command[command.index('--cpus-per-task')+1]=='1'
    assert '--no-requeue' in command


def test_lost_job_id_is_recovered_without_duplicate_submission(project,monkeypatch,tmp_path):
    _,cfg=project;root=tmp_path/'native'
    slurm.create_request(cfg,[],root,'digest')
    submit=Mock(return_value='response lost\n');monkeypatch.setattr(control.subprocess,'check_output',submit)
    with pytest.raises(RuntimeError,match='unambiguous'):control.submit(root)
    monkeypatch.setattr(control,'recover',lambda record:'124')
    assert control.submit(root)['job_id']=='124'
    assert submit.call_count==1


def test_uncertain_scheduler_history_never_resubmits(project,monkeypatch,tmp_path):
    _,cfg=project;root=tmp_path/'native'
    slurm.create_request(cfg,[],root,'digest')
    control.save(root/'submission.json',dict(launching=True,job_id=None,job_name='combichem-test',started_at=0))
    monkeypatch.setattr(control,'recover',Mock(side_effect=RuntimeError('accounting unavailable')))
    submit=Mock();monkeypatch.setattr(control.subprocess,'check_output',submit)
    with pytest.raises(RuntimeError,match='unavailable'):control.submit(root)
    submit.assert_not_called()
    assert json.loads((root/'submission.json').read_text())['launching']


def test_native_completion_alone_does_not_establish_workflow_completion(project,monkeypatch,tmp_path):
    _,cfg=project
    root=tmp_path/'native';request=slurm.create_request(cfg,[],root,'digest')
    monkeypatch.setattr(slurm,'_select_allocation',lambda *args:(root,request,'123'))
    monkeypatch.setattr(slurm,'_await_ready',lambda *args:None)
    finish=Mock();monkeypatch.setattr(slurm,'_finish_allocation',finish)
    with pytest.raises(RuntimeError,match='workflow result'):
        with slurm.allocation(cfg,flags()):pass
    finish.assert_called_once()
    assert not (root/'completed.json').exists()


def test_changed_input_never_replaces_an_active_allocation(project,monkeypatch):
    _,cfg=project
    directory=Path(cfg['out-dir'])/'slurm';directory.mkdir(parents=True)
    root=directory/'run-test';slurm.create_request(cfg,[],root,'different input')
    slurm.atomic_save(directory/'current.json',dict(session=root.name))
    monkeypatch.setattr(slurm,'control',lambda cfg,action,value:dict(job_id='123') if action=='submit'
                        else dict(active=True,state='RUNNING'))
    with pytest.raises(RuntimeError,match='different inputs'):slurm._select_allocation(cfg,[],slurm.signature(cfg,[]),directory)
    assert json.loads((directory/'current.json').read_text())['session']=='run-test'


def test_slurm_export_does_not_submit(project,monkeypatch):
    _,cfg=project;ns=flags();ns.export_only=True
    submit=Mock();monkeypatch.setattr(slurm,'control',submit)
    assert not slurm.dispatch(cfg,ns)
    with slurm.allocation(cfg,ns) as session:assert session is None
    submit.assert_not_called()


def test_node_pool_transport_preserves_pose_and_never_opens_sqlite(project,monkeypatch,tmp_path):
    from molnova import database
    from molnova.stages.combichem_slurm_pool import main
    from molnova.chemistry.combichem3d import history
    from molnova.chemistry.combichem3d.assembly import assemble
    from molnova.chemistry.combichem3d.clashes import BumpChecker,load_receptor
    from molnova.chemistry.combichem3d.output import versions
    from molnova.chemistry.combichem3d.search import search
    _,cfg=project
    cfg.update({'runtime-versions':versions(),'conformers':1,'torsion-trials':0,'slurm-runtime-dir':str(tmp_path),'slurm-job-id':'42'})
    parent,h=history.load_parents(cfg['parents'],cfg['history'])[0]
    product,birth=assemble(parent,h,h['attachments'],['*C','*CC'],'native-test')
    local=search(product,birth,BumpChecker(load_receptor(cfg['receptor'],cfg),cfg),cfg,37)
    monkeypatch.setenv('SLURM_JOB_ID','42');monkeypatch.setenv('SLURMD_NODENAME',socket.gethostname())
    pools=tmp_path/'pools';root=pools/socket.gethostname();root.mkdir(parents=True);(root/'token').write_text('private-test-token')
    monkeypatch.setattr(database,'connect',Mock(side_effect=AssertionError('Pure pool must not access SQLite')))
    process=multiprocessing.get_context('fork').Process(target=main,args=(pools,1))
    process.start();remote=slurm.SlurmSearch(socket.gethostname(),cfg)
    try:
        until=time.monotonic()+10
        while not (root/'endpoint.json').exists():
            assert process.is_alive() and time.monotonic()<until
            time.sleep(.05)
        endpoint=json.loads((root/'endpoint.json').read_text())
        with socket.create_connection((endpoint['host'],endpoint['port'])) as rejected:
            rejected.sendall(b'wrong-token\n')
            assert rejected.recv(1)==b''
        result=remote(product,birth,None,cfg,37)
        assert local['status']==result['status']=='PASS'
        np.testing.assert_array_equal(local['pose'].GetConformer().GetPositions(),result['pose'].GetConformer().GetPositions())
        assert history.snapshot(local['pose'],birth)==history.snapshot(result['pose'],birth)
    finally:
        remote.close();(root/'stop').touch();process.join(15)
        if process.is_alive():process.terminate();process.join()
    assert process.exitcode==0


def test_slurm_coordinator_relaunch_is_local_not_ssh(project,monkeypatch):
    _,cfg=project
    monkeypatch.delenv('MOLNOVA_COMBICHEM_COORDINATOR',raising=False)
    run=Mock();monkeypatch.setattr(slurm.subprocess,'run',run)
    assert slurm.dispatch(cfg,flags())
    command=run.call_args.args[0]
    assert command[:2]==['bash','-c'] and 'MOLNOVA_COMBICHEM_COORDINATOR=1' in command[2]
    assert 'ssh ' not in command[2]
    monkeypatch.setenv('MOLNOVA_COMBICHEM_COORDINATOR','1')
    assert not slurm.dispatch(cfg,flags())


def test_database_is_initialized_before_scheduler_metadata(project,monkeypatch):
    from contextlib import contextmanager
    from molnova.stages import combichem_3d as stage
    filename,cfg=project
    monkeypatch.setenv('MOLNOVA_COMBICHEM_COORDINATOR','1')
    @contextmanager
    def allocated(config,ns):
        assert (Path(config['out-dir'])/'combichem3d.sqlite').is_file()
        raise RuntimeError('allocation reached after valid project initialization')
        yield
    monkeypatch.setattr(slurm,'allocation',allocated)
    with pytest.raises(RuntimeError,match='allocation reached'):
        stage._main([str(filename)])
