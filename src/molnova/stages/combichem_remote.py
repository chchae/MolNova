"""Direct SSH launch of the complete COMBICHEM supervisor on shared storage."""
import os
import hashlib
from pathlib import Path
import shlex
import socket
import subprocess


def dispatch(cfg, ns):
    from molnova.stages import combichem_3d as stage
    from molnova.stages.combichem_distributed import install, uninstall
    if cfg.get('worker-hosts'):
        # The main stage fingerprints chemistry and itself. Include the SSH
        # calculation adapter too, without changing legacy local manifests.
        names = ['combichem_remote.py','combichem_distributed.py','combichem_compute.py']
        if cfg.get('execution-backend')=='slurm':
            names += ['combichem_slurm.py','combichem_slurm_pool.py','combichem_slurm_control.py','combichem_slurm_worker.py','combichem_product.py']
        cfg['remote-implementation-sha256'] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in names
        }
        install(stage)
    else:
        uninstall(stage)
    if cfg.get('execution-backend')=='slurm':
        from molnova.stages.combichem_slurm import dispatch as slurm_dispatch
        return slurm_dispatch(cfg,ns)
    host = cfg.get('execution-host')
    if not host:
        return False
    local = socket.gethostname().split('.')[0]
    if local == host.rsplit('@', 1)[-1].split('.')[0]:
        return False
    if os.environ.get('MOLNOVA_COMBICHEM_REMOTE') == '1':
        # SSH aliases need not match the physical hostname. This marker is set
        # only for the already-dispatched command, so never launch recursively.
        return False
    src = str(Path(__file__).resolve().parents[2])
    args = ['python', '-u', '-m', 'molnova.cli', 'combichem-3d', cfg['config-path']]
    for flag in ('export_only', 'debug', 'trace'):
        if getattr(ns, flag):
            args.append('--' + flag.replace('_', '-'))
    body = (
        'set -e; source ' + shlex.quote(cfg['remote-env-script']) + '; '
        # Do not leak Amber Python 3.12 site-packages into Schrödinger Python.
        'export PYTHONPATH=' + shlex.quote(src) + '; '
        'export MOLNOVA_COMBICHEM_REMOTE=1; '
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
        'cd ' + shlex.quote(str(Path(cfg['config-path']).parent)) + '; '
        'exec ' + shlex.join(args)
    )
    print(f'[combichem-3d] forwarding execution to {host} (direct SSH, shared paths)', flush=True)
    subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    host, 'bash -c ' + shlex.quote(body)], check=True)
    return True
