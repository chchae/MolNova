"""Synthetic node calculations and multiplexed local persistence, without Slurm submission."""
from contextlib import contextmanager
import importlib.util
import json
import multiprocessing
from pathlib import Path
import socket
import sqlite3
import time
import threading
from unittest.mock import Mock

import pytest
from molnova import database
from molnova.config import combichem_3d_settings
from molnova.stages import combichem_3d as stage
from molnova.stages import combichem_product as product
from molnova.stages import combichem_slurm as slurm
from molnova.stages import combichem_slurm_worker as worker
from molnova.chemistry.combichem3d.clashes import BumpChecker,load_receptor
from molnova.chemistry.combichem3d.evaluation import Evaluator
from molnova.chemistry.combichem3d.frames import ReferencePoses


@pytest.fixture
def project(tmp_path):
    path=Path(__file__).parents[1]/'examples/combichem-3d/make_toy_inputs.py'
    spec=importlib.util.spec_from_file_location('pipeline_toy',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    filename=module.create(tmp_path/'toy',1,2)
    cfg=combichem_3d_settings(filename)
    cfg.update({'growth-mode':'frame-cycles','reference-mcs':'c1([*:0])c([*:1])cccc1',
                'stop-mode':'cycles','max-iteration':2,'sidechains-per-site':2,
                'sidechain-include-h':False,'conformers':1,'torsion-trials':0,
                'max-seconds':60,'processes':2,'coordinator-processes':1,
                'execution-backend':'slurm','worker-hosts':{socket.gethostname():2},
                'batch-size':2,'result-buffer-size':2,'slurm-worker-index':0})
    return filename,cfg


@contextmanager
def node(cfg, tmp_path, monkeypatch):
    from molnova.stages.combichem_slurm_pool import main
    root=tmp_path/'pools'/socket.gethostname();root.mkdir(parents=True)
    (root/'token').write_text('test-token')
    cfg.update({'slurm-runtime-dir':str(tmp_path),'slurm-job-id':'42'})
    monkeypatch.setenv('SLURM_JOB_ID','42');monkeypatch.setenv('SLURMD_NODENAME',socket.gethostname())
    with monkeypatch.context() as patch:
        deny=Mock(side_effect=AssertionError('Node must never open SQLite'))
        patch.setattr(database,'connect',deny);patch.setattr(sqlite3,'connect',deny)
        process=multiprocessing.get_context('fork').Process(target=main,args=(tmp_path/'pools',2))
        process.start()
    try:
        until=time.monotonic()+10
        while not (root/'endpoint.json').exists():
            assert process.is_alive() and time.monotonic()<until
            time.sleep(.02)
        yield
    finally:
        (root/'stop').touch();process.join(15)
        if process.is_alive():process.terminate();process.join()
    assert process.exitcode==0


def test_slot_partition_covers_all_384_slots_without_384_local_processes():
    cfg={'worker-hosts':{'cpu1':192,'cpu2':192},'coordinator-processes':16}
    groups=[worker.assigned_hosts(cfg,index) for index in range(16)]
    assert all(len(group)==24 for group in groups)
    assert sum(group.count('cpu1') for group in groups)==192
    assert sum(group.count('cpu2') for group in groups)==192


@pytest.mark.parametrize('count',[0,True,3,1.5])
def test_invalid_coordinator_count_is_rejected(project,count):
    filename,_=project
    filename.write_text(filename.read_text().replace('[filters]',
        f'coordinator-processes = {str(count).lower()}\n[filters]'))
    with pytest.raises(ValueError,match='coordinator-processes'):combichem_3d_settings(filename)


def test_node_complete_products_match_local_calculation_exactly(project,tmp_path,monkeypatch):
    _,cfg=project
    db,_=stage.prepare_project(cfg)
    row=database.growth_parent(db,1)
    plan={'sites':row['history']['attachments'],'pools':[['*C','*CC'],['*C','*CC']]}
    parent=stage.restore_parent(row)
    checker=BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
    evaluator=Evaluator(cfg['reference-smiles'],cfg['filters'])
    references=ReferencePoses(cfg['reference-poses'],cfg['reference-mcs'])
    deadline=time.time()+60
    expected=[product.evaluate_product(parent,row['history'],plan,i,cfg,checker,evaluator,deadline,references)
              for i in range(4)]
    with node(cfg,tmp_path,monkeypatch):
        remote=slurm.SlurmSearch(socket.gethostname(),cfg)
        try:
            received=[]
            result=remote.exchange(dict(kind='batch',molblock=row['molblock'],history=row['history'],plan=plan,
                cursor=0,stop=4,deadline=deadline),product=received.append)
            assert result=={'batch_done':True}
            assert [r['_product'] for r in received]==list(range(4))
            # Compare full reports, coordinates, UID/branch history and descriptors.
            assert [(r['report'],r['payload']) for r in received]==expected
            assert all(payload and payload['history']['constraint_reference'] for _,payload in expected)
        finally:remote.close()


def test_multiplexed_worker_runs_two_frame_cycles_without_local_product_work(project,tmp_path,monkeypatch):
    _,cfg=project
    db,first=stage.prepare_project(cfg)
    with node(cfg,tmp_path,monkeypatch):
        monkeypatch.setattr(product,'evaluate_product',Mock(side_effect=AssertionError('Products must run on node')))
        monkeypatch.setattr(stage,'BumpChecker',Mock(side_effect=AssertionError('No local receptor setup')))
        original=slurm.SlurmSearch.exchange
        barrier=threading.Barrier(2)
        lock=threading.Lock()
        counts={'started':0,'active':0,'peak':0}
        def observed(self,request,*args,**kwargs):
            if request.get('kind')!='batch':return original(self,request,*args,**kwargs)
            with lock:
                counts['started']+=1;counts['active']+=1
                counts['peak']=max(counts['peak'],counts['active'])
                first=counts['started']<=2
            try:
                if first:barrier.wait(timeout=5)
                return original(self,request,*args,**kwargs)
            finally:
                with lock:counts['active']-=1
        monkeypatch.setattr(slurm.SlurmSearch,'exchange',observed)
        database.growth_start_session(db,cfg)
        with database.stage_lock(db,'combichem3d'):
            for iteration in range(first,cfg['max-iteration']+1):
                worker.worker_loop(db,cfg,iteration)
        info=database.growth_run_info(db)
        assert counts['peak']==2
        assert info['attempted']>=4 and info['unique_count']>0
        assert len(database.growth_rows(db,'combichem3d_cycle'))==2
        batches=database.growth_rows(db,'combichem3d_batch')
        assert batches and all(row['state']=='done' for row in batches)
        assert not list((Path(db).parent/'result-buffer').glob('*.jsonl'))
        assert not any(event['kind']=='worker_error' for event in database.growth_rows(db,'combichem3d_event'))


@pytest.mark.parametrize('fault',['disconnect','missing_completion','persistence'])
def test_received_results_survive_failed_batch_and_resume(project,tmp_path,monkeypatch,fault):
    _,cfg=project
    cfg['max-iteration']=1
    db,first=stage.prepare_project(cfg)
    with node(cfg,tmp_path,monkeypatch):
        database.growth_start_session(db,cfg)
        with database.stage_lock(db,'combichem3d'):
            worker.worker_loop(db,cfg,first)  # initial input bookkeeping
            with monkeypatch.context() as patch:
                if fault=='persistence':
                    save=database.growth_save_attempts
                    patch.setattr(database,'growth_save_attempts',Mock(side_effect=OSError('storage unavailable')))
                else:
                    original=slurm.SlurmSearch.exchange
                    lock=threading.Lock()
                    chosen=[False]
                    def broken(self,request,*args,**kwargs):
                        if request.get('kind')!='batch':return original(self,request,*args,**kwargs)
                        with lock:
                            fail=not chosen[0];chosen[0]=True
                        if fail and fault=='disconnect':
                            receive=kwargs['product']
                            def interrupted(value):
                                receive(value)
                                raise slurm.RemoteCalculationError('injected disconnect')
                            kwargs['product']=interrupted
                        reply=original(self,request,*args,**kwargs)
                        return {} if fail and fault=='missing_completion' else reply
                    patch.setattr(slurm.SlurmSearch,'exchange',broken)
                with pytest.raises((slurm.RemoteCalculationError,OSError)):
                    worker.worker_loop(db,cfg,first+1)
            assert database.growth_run_info(db)['stop_reason']=='worker_error'
            if fault=='persistence':
                journals=list((Path(db).parent/'result-buffer').glob('*.jsonl'))
                assert journals
                from molnova.stages.combichem_buffer import recover
                recover(db)
            saved=database.growth_run_info(db)['attempted']
            assert saved== (3 if fault=='disconnect' else 4)
            # Re-arm under the exclusive stage lock only, retaining saved cursors.
            database.growth_end_session(db,'worker_error')
            database.growth_start_session(db,cfg)
            worker.worker_loop(db,cfg,first+1)
            assert database.growth_run_info(db)['attempted']==4
            rows=database.growth_rows(db,'combichem3d_attempt')
            assert len({(r['parent_pose'],r['ordinal']) for r in rows})==4
            assert all(r['state']=='done' for r in database.growth_rows(db,'combichem3d_batch'))


def test_supervisor_uses_local_owner_count_and_preserves_inherited_stage_lock(project,tmp_path,monkeypatch):
    from molnova.stages import combichem_distributed as distributed
    _,cfg=project
    cfg['max-iteration']=1
    db,first=stage.prepare_project(cfg)
    with node(cfg,tmp_path,monkeypatch):
        monkeypatch.setattr(stage,'worker_loop',distributed.worker_loop)
        monkeypatch.setattr(product,'evaluate_product',Mock(side_effect=AssertionError('No local product work')))
        monkeypatch.setattr(stage,'BumpChecker',Mock(side_effect=AssertionError('No local receptor setup')))
        with database.stage_lock(db,'combichem3d'):
            stage.supervise(cfg,db,first)
        info=database.growth_run_info(db)
        assert info['attempted']==4 and info['stop_reason']=='exhausted'
        batches=database.growth_rows(db,'combichem3d_batch')
        assert len({r['owner'] for r in batches})==1
        assert all(r['state']=='done' for r in batches)


def test_expired_batch_deadline_preserves_uncomputed_cursor(project,tmp_path,monkeypatch):
    _,cfg=project
    db,_=stage.prepare_project(cfg)
    row=database.growth_parent(db,1)
    plan={'sites':row['history']['attachments'],'pools':[['*C','*CC'],['*C','*CC']]}
    with node(cfg,tmp_path,monkeypatch):
        remote=slurm.SlurmSearch(socket.gethostname(),cfg)
        try:
            received=[]
            result=remote.exchange(dict(kind='batch',molblock=row['molblock'],history=row['history'],plan=plan,
                cursor=1,stop=4,deadline=time.time()-1),product=received.append)
            assert result=={'batch_interrupted':1} and received==[]
        finally:remote.close()
