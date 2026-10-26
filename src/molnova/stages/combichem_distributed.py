"""SQLite workers stay on one host; SSH helpers perform pure search only."""
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import logging
from rdkit import Chem
from molnova.chemistry.combichem3d import history as provenance

_ORIGINAL_WORKER = None


def install(stage):
    global _ORIGINAL_WORKER
    if _ORIGINAL_WORKER is None:
        _ORIGINAL_WORKER = stage.worker_loop
    stage.worker_loop = worker_loop


def uninstall(stage):
    if _ORIGINAL_WORKER is not None:
        stage.worker_loop = _ORIGINAL_WORKER


class RemoteCalculationError(OSError):
    """Transport failure is not a scientific UNRESOLVED result."""


class RemoteSearch:
    def __init__(self, host, cfg):
        self.host, self.cfg, self.process = host, cfg, None

    def start(self):
        src = str(Path(__file__).resolve().parents[2])
        script = ('set -e; source ' + shlex.quote(self.cfg['remote-env-script']) + '; '
                  'export PYTHONPATH=' + shlex.quote(src) + '; '
                  'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
                  'exec python -u -m molnova.stages.combichem_compute')
        # Bound simultaneous SSH authentication starts (sshd MaxStartups), while
        # keeping independent persistent connections for all computation slots.
        gate = Path(tempfile.gettempdir()) / f'molnova-combichem-ssh-{os.getuid()}-{self.host}.lock'
        with gate.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            options = []
            if self.cfg.get('worker-ssh-known-hosts'):
                options = ['-o', 'StrictHostKeyChecking=yes', '-o',
                           'UserKnownHostsFile=' + self.cfg['worker-ssh-known-hosts']]
            self.process = subprocess.Popen(
                ['ssh', *options, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                 # Each calculator needs its own session. A shared master hits
                 # sshd's per-connection session cap; ControlMaster=no alone
                 # still permits reuse, so explicitly disable ControlPath too.
                 '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ControlPersist=no',
                 '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
                 self.host, 'bash -c ' + shlex.quote(script)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self.process.stdin.write(json.dumps(self.cfg) + '\n')
            self.process.stdin.flush()
            if self.process.stdout.readline().strip() != '{"ready": true}':
                raise RemoteCalculationError(f'Remote calculator initialization failed on {self.host}')
            logging.getLogger('molnova.combichem3d').info('remote calculator ready host=%s', self.host)

    def __call__(self, product, history, checker, cfg, seed, *, deadline=float('inf')):
        try:
            return self.calculate(product, history, seed, deadline)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RemoteCalculationError(
                f'Remote calculator failed on {self.host}; uncommitted claim retained: {exc}') from exc

    def calculate(self, product, history, seed, deadline):
        if self.process is None:
            self.start()
        # Molblocks round coordinates to four decimals. The history carries the
        # exact current coordinates, including new atoms, in both directions.
        request = dict(molblock=Chem.MolToMolBlock(product),
                       history=provenance.snapshot(product, history), seed=seed, deadline=deadline)
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RemoteCalculationError(f'Remote calculator disconnected on {self.host}')
        result = json.loads(line)
        if 'molblock' in result:
            pose = Chem.MolFromMolBlock(result.pop('molblock'), removeHs=False)
            if pose is None:
                raise ValueError('Invalid remote calculator pose')
            result['pose'] = provenance.restore(pose, dict(history, pose_coordinates=result.pop('coordinates')))
        return result

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
            self.process.stdout.close()


def worker_loop(db, cfg, iteration):
    from molnova.stages import combichem_3d as stage
    hosts = [host for host, slots in cfg['worker-hosts'].items() for _ in range(slots)]
    index = multiprocessing.current_process()._identity[-1] - 1
    host = hosts[index % len(hosts)]
    local = socket.gethostname().split('.')[0]
    remote = None
    logging.getLogger('molnova.combichem3d').info('calculator slot host=%s SQLite host=%s', host, local)
    if cfg.get('execution-backend')=='slurm' or host.rsplit('@', 1)[-1].split('.')[0] != local:
        if cfg.get('execution-backend')=='slurm':
            from molnova.stages.combichem_slurm import SlurmSearch
            remote = SlurmSearch(host, cfg)
            from molnova.chemistry.combichem3d import cycles
            stage.methyl_probe = remote.probe
            cycles.methyl_probe = remote.probe
            cycles.select_mcs = remote.select_mcs
        else:
            remote = RemoteSearch(host, cfg)
        stage.search = remote
    try:
        _ORIGINAL_WORKER(db, cfg, iteration)
    finally:
        if remote is not None:
            remote.close()
