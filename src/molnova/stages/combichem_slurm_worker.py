"""Bounded local claim owners multiplex pure Slurm batches; only owners write DB."""
from concurrent.futures import ThreadPoolExecutor
import os
from queue import Queue
import signal
import time
import traceback

from rdkit import Chem

from molnova import database
from molnova.chemistry.combichem3d.logging_utils import logger
from molnova.stages.combichem_buffer import ResultBuffer
from molnova.stages.combichem_slurm import SlurmSearch, RemoteCalculationError


def assigned_hosts(cfg, index):
    hosts = [host for host,count in cfg['worker-hosts'].items() for _ in range(count)]
    return hosts[index::cfg['coordinator-processes']]


class NodeEvaluation:
    """Small planning/input requests share an idle calculator, never local RDKit work."""
    def __init__(self, remote):
        self.remote = remote

    def request(self, kind, mol):
        return self.remote.exchange(dict(kind=kind,molblock=Chem.MolToMolBlock(mol),
            coordinates=mol.GetConformer().GetPositions().tolist()))['value']

    def check(self, mol):
        return self.request('baseline',mol)

    def evaluate(self, mol):
        return self.request('properties',mol)


def worker_loop(db, cfg, iteration):
    from molnova.stages import combichem_3d as stage
    from molnova.chemistry.combichem3d import cycles
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,signal.SIG_DFL)
    hosts = assigned_hosts(cfg,cfg['slurm-worker-index'])
    clients = [SlurmSearch(host,cfg) for host in hosts]
    context = NodeEvaluation(clients[0])
    originals = stage.methyl_probe,cycles.methyl_probe,cycles.select_mcs
    stage.methyl_probe = clients[0].probe
    cycles.methyl_probe = clients[0].probe
    cycles.select_mcs = clients[0].select_mcs
    events = Queue(maxsize=max(2,2*len(clients)))
    active, buffers = {}, {}
    available = list(range(len(clients)))
    deferred = None
    failure = None
    library = None
    deadline = database.growth_run_info(db)['deadline']

    def calculate(slot, job, row):
        try:
            request = dict(kind='batch',molblock=row['molblock'],history=row['history'],plan=row['plan'],
                           cursor=int(job['cursor']),stop=int(job['stop']),deadline=deadline)
            next_ordinal = request['cursor']
            def receive(value):
                nonlocal next_ordinal
                if value['_product'] != next_ordinal or next_ordinal >= request['stop']:
                    raise RemoteCalculationError('Native batch ordinal mismatch; claim retained')
                next_ordinal += 1
                events.put(('product',slot,value))
            result = clients[slot].exchange(request,product=receive)
            complete = result == {'batch_done':True} and next_ordinal == request['stop']
            timed_out = (result == {'batch_interrupted':next_ordinal}
                         and next_ordinal < request['stop'] and time.time() >= deadline)
            if not complete and not timed_out:
                raise RemoteCalculationError('Missing native batch completion; claim retained')
        except BaseException as exc:
            events.put(('error',slot,exc))
        else:
            events.put(('done',slot,None))

    def bookkeeping(job):
        nonlocal library
        if job['kind']=='input':
            row = database.growth_parent(db,job['parent_pose'])
            parent = stage.restore_parent(row)
            properties,passed,rejected = context.evaluate(parent)
            baseline_pose = Chem.AddHs(parent,addCoords=True)
            baseline = context.check(baseline_pose)
            baseline.update(kind='baseline',reason='Input pose before growth; reported separately, not silently subtracted',
                            filter_rejections=rejected,generated_hydrogens=baseline_pose.GetNumAtoms()-parent.GetNumAtoms())
            database.growth_save_input(db,job['parent_pose'],properties,passed and baseline['passed'],baseline,
                prune=cfg['prune-intermediates'] and not passed,
                pose_data=dict(molblock=Chem.MolToMolBlock(baseline_pose),
                               history=stage.provenance.snapshot(baseline_pose,row['history'])))
        elif job['kind']=='frame':
            cycles.prepare_next_frame(db,iteration,cfg,context,deadline)
        elif job['kind']=='plan':
            if library is None:library,_ = stage.load_library(cfg['sidechain-library'],cfg)
            stage.plan_parent(db,job,cfg,library,context,deadline)
        else:
            raise ValueError(f'Unexpected planning job: {job["kind"]}')

    logger.info('Slurm local claim owner=%s calculator_slots=%s hosts=%s',os.getpid(),len(clients),hosts)
    try:
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            try:
                while True:
                    stopping = failure is not None or database.growth_should_stop(db)
                    if deferred is not None and not active and not stopping:
                        bookkeeping(deferred)
                        deferred = None
                    exhausted = False
                    while available and deferred is None and not stopping:
                        job = database.growth_claim(db,iteration,cfg['batch-size'],os.getpid())
                        if job is None:
                            exhausted = True
                            break
                        if job['kind']=='wait':break
                        if job['kind'] in ('input','frame','plan'):
                            deferred = job
                            break
                        slot = available.pop()
                        row = database.growth_parent(db,job['parent_pose'])
                        buffer = ResultBuffer(db,job,cfg)
                        buffer.__enter__()
                        executor.submit(calculate,slot,job,row)
                        active[slot] = job
                        buffers[slot] = buffer
                    if not active:
                        if stopping or exhausted:break
                        if deferred is None:time.sleep(.05)
                        continue
                    kind,slot,value = events.get()
                    if kind=='product':
                        buffers[slot].append(value['_product'],value['report'],value['payload'])
                    else:
                        buffer,job = buffers.pop(slot),active.pop(slot)
                        buffer.__exit__(None,None,None)
                        if kind=='done':database.growth_finish_batch(db,job)
                        else:
                            failure = failure or value
                            database.growth_request_stop(db,'worker_error')
                        available.append(slot)
            except BaseException:
                # Drain network readers before executor shutdown. A full event
                # queue must never deadlock shutdown or discard received poses.
                try:
                    database.growth_request_stop(db,'worker_error')
                except Exception:
                    logger.exception('Cannot persist stop request; draining received results')
                while active:
                    kind,slot,value = events.get()
                    try:
                        if kind=='product':
                            buffers[slot].append(value['_product'],value['report'],value['payload'])
                        else:
                            buffers[slot].__exit__(None,None,None)
                            if kind=='done':database.growth_finish_batch(db,active[slot])
                    except Exception:
                        logger.exception('Result persistence failed while draining; durable journal retained')
                    finally:
                        if kind!='product':del buffers[slot],active[slot]
                raise
        if failure is not None:raise failure
    except BaseException:
        logger.exception('Slurm local claim owner failed; claims/results retained')
        try:
            database.growth_log_event(db,'worker_error',dict(reason=traceback.format_exc(),owner=os.getpid()))
            database.growth_request_stop(db,'worker_error')
        except Exception:
            logger.exception('Cannot persist worker diagnostic; claims/journals retained')
        raise
    finally:
        stage.methyl_probe,cycles.methyl_probe,cycles.select_mcs = originals
        for client in clients:client.close()
