"""Local SQLite coordination with Slurm-only pure calculation node pools."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import shlex
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from rdkit import Chem
from molnova.chemistry.combichem3d import history as provenance
from molnova.chemistry.combichem3d.logging_utils import logger
from molnova.stages.combichem_distributed import RemoteSearch, RemoteCalculationError


def atomic_save(path, value):
    pending = path.with_suffix('.pending')
    with pending.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def shell_environment(cfg):
    src = str(Path(__file__).resolve().parents[2])
    return ('set -e; source '+shlex.quote(cfg['remote-env-script'])+'; '
            'export PYTHONPATH='+shlex.quote(src)+'; '
            'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; ')


def control(cfg, action, value):
    helper = str(Path(__file__).with_name('combichem_slurm_control.py'))
    command = shlex.join(['python3', helper, action, str(value)])
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             cfg['slurm-host'], command], check=True, capture_output=True,
                            text=True, timeout=125)
    return json.loads(result.stdout)


def signature(cfg, flags):
    sources = list(Path(__file__).resolve().parent.glob('combichem*.py'))
    sources += list(Path(__file__).resolve().parents[1].joinpath('chemistry/combichem3d').glob('*.py'))
    sources += [Path(__file__).resolve().parents[1]/name for name in ('config.py','database.py','amber_slurm_control.py')]
    inputs = [Path(cfg['config-path'])]
    inputs += [Path(cfg[key]) for key in ('parents','history','receptor','sidechain-library',
               'reference-smiles','reference-source-poses') if cfg.get(key)
               and not Path(cfg[key]).is_relative_to(Path(cfg['out-dir']))]
    files = {str(path):hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sources+inputs if path.is_file()}
    source_cfg = {k:v for k,v in cfg.items() if k not in ('slurm-runtime-dir','slurm-job-id')}
    return hashlib.sha256(json.dumps(dict(cfg=source_cfg, flags=flags, files=files), sort_keys=True).encode()).hexdigest()


def create_request(cfg, flags, root, digest):
    root.mkdir(mode=0o700)
    token = uuid.uuid4().hex
    hours, rest = divmod(cfg['slurm-time-limit'],3600); minutes, seconds = divmod(rest,60)
    request = dict(signature=digest, token=token, job_name='combichem-'+token,
                   config_path=cfg['config-path'], flags=flags, hosts=cfg['worker-hosts'],
                   partition=cfg['slurm-partition'], memory_mb=cfg['slurm-memory-mb'],
                   time_limit=f'{hours:02}:{minutes:02}:{seconds:02}')
    atomic_save(root/'request.json', request)
    for host in cfg['worker-hosts']:
        directory = root/'pools'/host
        directory.mkdir(parents=True,mode=0o700)
        (directory/'token').write_text(secrets.token_hex(32))
    command = ['python','-u','-m','molnova.stages.combichem_slurm','launch',str(root)]
    (root/'run.sbatch').write_text('#!/bin/bash\n'+shell_environment(cfg)+'exec '+shlex.join(command)+'\n')
    return request


def dispatch(cfg, ns):
    """Activate a compatible calculator environment locally, never SSH to a node."""
    if ns.export_only or os.environ.get('MOLNOVA_COMBICHEM_COORDINATOR')=='1':
        return False
    command = ['python','-u','-m','molnova.cli','combichem-3d',cfg['config-path']]
    command += ['--'+name.replace('_','-') for name in ('debug','trace') if getattr(ns,name)]
    body = shell_environment(cfg)+'export MOLNOVA_COMBICHEM_COORDINATOR=1; exec '+shlex.join(command)
    print('[combichem-3d] activating calculator environment on current coordinator '+socket.gethostname(),flush=True)
    subprocess.run(['bash','-c',body],check=True)
    return True


def _select_allocation(cfg, flags, digest, directory):
    current = directory/'current.json'
    root = None
    if current.exists():
        name = json.loads(current.read_text())['session']
        if Path(name).name!=name or not name.startswith('run-'):
            raise ValueError('Invalid saved Slurm session directory')
        root = directory/name
        request = json.loads((root/'request.json').read_text())
        submitted = control(cfg,'submit',root)
        if submitted.get('busy'):
            raise RuntimeError('Slurm submission is still registering; retry to reattach without resubmission')
        state = control(cfg,'status',submitted['job_id'])
        if state['active']:
            if request['signature']!=digest:
                raise RuntimeError('An existing Slurm allocation is active with different inputs; retained')
        else:
            # Pools alone do not establish workflow completion. A fresh native
            # allocation may resume SQLite only after the old one is terminal.
            root = None
    if root is None:
        session = 'run-'+uuid.uuid4().hex
        root = directory/session
        request = create_request(cfg,flags,root,digest)
        atomic_save(current,dict(session=session))
        submitted = control(cfg,'submit',root)
        while submitted.get('busy'):
            time.sleep(cfg['slurm-poll-seconds'])
            submitted = control(cfg,'submit',root)
    return root,request,submitted['job_id']


def _await_ready(cfg, root, job_id):
    from molnova.chemistry.combichem3d.output import versions
    previous = None
    while True:
        state = control(cfg,'status',job_id)
        if state['state']!=previous:
            logger.info('Slurm JobId=%s state=%s',job_id,state['state'])
            previous = state['state']
        if not state['active']:
            raise RuntimeError(f'Slurm calculator allocation {job_id} ended before workflow: {state["state"]}')
        paths = [root/'pools'/host/'endpoint.json' for host in cfg['worker-hosts']]
        if state['state']=='RUNNING' and all(path.exists() for path in paths):
            for host,path in zip(cfg['worker-hosts'],paths):
                endpoint = json.loads(path.read_text())
                if (endpoint['host']!=host or endpoint['job_id']!=job_id
                        or endpoint['slots']!=cfg['worker-hosts'][host] or endpoint['versions']!=versions()):
                    raise RuntimeError('Slurm calculator node/CPU/version identity mismatch')
            return
        time.sleep(cfg['slurm-poll-seconds'])


def _finish_allocation(cfg, root, job_id):
    for host in cfg['worker-hosts']:
        (root/'pools'/host/'stop').touch()
    until = time.monotonic()+60
    while True:
        state = control(cfg,'status',job_id)
        if not state['active']:
            if not state.get('successful'):
                raise RuntimeError(f'Slurm calculator allocation {job_id} failed: {state["state"]}; results retained')
            return
        if time.monotonic()>until:
            control(cfg,'cancel',job_id)
            raise RuntimeError('Slurm calculator shutdown timed out; own allocation cancellation requested, results retained')
        time.sleep(cfg['slurm-poll-seconds'])


@contextmanager
def allocation(cfg, ns):
    """Called under the local growth stage lock, before claims/session recovery."""
    if cfg.get('execution-backend')!='slurm' or ns.export_only:
        yield None
        return
    flags = ['--'+name.replace('_','-') for name in ('debug','trace') if getattr(ns,name)]
    digest = signature(cfg,flags)
    directory = Path(cfg['out-dir'])/'slurm';directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    root,request,job_id = _select_allocation(cfg,flags,digest,directory)
    logger.info('Slurm JobId=%s coordinator=%s hosts=%s output=%s/job.out',
                job_id,socket.gethostname(),cfg['worker-hosts'],root)
    _await_ready(cfg,root,job_id)
    if signature(cfg,flags)!=digest:
        raise RuntimeError('Inputs/source changed while waiting for Slurm; allocation retained')
    cfg.update({'slurm-runtime-dir':str(root),'slurm-job-id':job_id})
    session = SimpleNamespace(result=None)
    try:
        yield session
    finally:
        _finish_allocation(cfg,root,job_id)
    if session.result is None:
        raise RuntimeError('No verified local workflow result; native completion alone is insufficient')
    info = session.result
    atomic_save(root/'completed.json',dict(token=request['token'],signature=digest,job_id=job_id,
                coordinator=socket.gethostname(),stop_reason=info['stop_reason'],
                attempted=info['attempted'],unique_count=info['unique_count']))


class SlurmSearch(RemoteSearch):
    def start(self):
        root = Path(self.cfg['slurm-runtime-dir'])/'pools'/self.host
        endpoint = json.loads((root/'endpoint.json').read_text())
        if endpoint['host']!=self.host or endpoint['job_id']!=self.cfg['slurm-job-id']:
            raise RemoteCalculationError('Slurm calculator endpoint belongs to another node/allocation')
        self.connection = socket.create_connection((endpoint['host'],endpoint['port']), timeout=30)
        self.connection.setsockopt(socket.SOL_SOCKET,socket.SO_KEEPALIVE,1)
        for option,value in (('TCP_KEEPIDLE',30),('TCP_KEEPINTVL',30),('TCP_KEEPCNT',3)):
            if hasattr(socket,option):
                self.connection.setsockopt(socket.IPPROTO_TCP,getattr(socket,option),value)
        stream = self.connection.makefile('rw',encoding='utf-8')
        self.process = SimpleNamespace(stdin=stream,stdout=stream)
        stream.write((root/'token').read_text().strip()+'\n')
        stream.write(json.dumps(self.cfg)+'\n');stream.flush()
        if stream.readline().strip()!='{"ready": true}':
            raise RemoteCalculationError('Slurm calculator initialization failed')
        self.connection.settimeout(None)

    def exchange(self, request, checkpoint=lambda report:None, product=lambda result:None):
        try:
            if self.process is None:self.start()
            self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:raise RemoteCalculationError('Slurm calculator disconnected; claim retained')
                result = json.loads(line)
                if '_product' in result:product(result)
                elif '_checkpoint' in result:checkpoint(result['_checkpoint'])
                elif '_log' in result:logger.log(result['level'],result['_log'])
                else:return result
        except (OSError,ValueError,KeyError,TypeError) as exc:
            raise RemoteCalculationError(f'Slurm calculator {self.host} failed; claim retained: {exc}') from exc

    def calculate(self, product, history, seed, deadline):
        result = self.exchange(dict(molblock=Chem.MolToMolBlock(product),history=provenance.snapshot(product,history),
                                    seed=seed,deadline=deadline))
        if 'molblock' in result:
            pose = Chem.MolFromMolBlock(result.pop('molblock'),removeHs=False)
            if pose is None:raise RemoteCalculationError('Invalid Slurm calculator pose')
            result['pose'] = provenance.restore(pose,dict(history,pose_coordinates=result.pop('coordinates')))
        return result

    def probe(self, parent, history, uid, checker, cfg, seed, deadline=float('inf')):
        return self.exchange(dict(kind='methyl_probe',molblock=Chem.MolToMolBlock(parent),
                history=provenance.snapshot(parent,history),uid=uid,seed=seed,deadline=deadline))

    def select_mcs(self, elites, previous, cfg, deadline, checkpoint=lambda report:None):
        result = self.exchange(dict(kind='mcs',elites=elites,previous=previous,deadline=deadline),checkpoint)
        selected = result['selected']
        if selected is not None:
            pose = Chem.MolFromMolBlock(selected['molblock'],removeHs=False)
            if pose is None:raise RemoteCalculationError('Invalid Slurm frame')
            selected = (provenance.restore(pose,selected['history']),selected['history'])
        return selected,result['report']

    def close(self):
        if self.process is not None:
            try:self.process.stdin.close()
            finally:
                self.connection.close()
                self.process = None


def launch(root):
    """Native batch driver supervises only pure calculator steps, never SQLite."""
    root = Path(root)
    request = json.loads((root/'request.json').read_text())
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id or not job_id.isdigit():raise RuntimeError('Native pool launch requires a Slurm allocation')
    nodes = subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split()
    slots = list(request['hosts'].values())[0]
    if set(nodes)!=set(request['hosts']) or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))<slots:
        raise RuntimeError('Slurm node/CPU allocation differs from the saved request')
    command = ['srun','--exact','--exclusive','--nodes',str(len(nodes)),
               '--ntasks',str(len(nodes)),'--ntasks-per-node=1','--cpus-per-task',str(slots),
               '--hint=nomultithread','--cpu-bind=cores','--kill-on-bad-exit=1','--export=ALL',
               sys.executable,'-u','-m','molnova.stages.combichem_slurm_pool',str(root/'pools'),str(slots)]
    subprocess.run(command,check=True)


if __name__=='__main__':
    if sys.argv[1]!='launch':raise ValueError('Expected launch')
    launch(sys.argv[2])
