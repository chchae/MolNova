"""Worker-owned shared-frame transport, Adam submission and committed cleanup."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from molnova import database, md_gbsa
from molnova.md_gbsa_native import save


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def add_settings(args, settings):
    if not getattr(args, 'md_gbsa_local_compute', False):
        return settings
    root = str(Path(args.md_gbsa_local_work_dir)/('molnova-'+Path(settings['root']).name))
    settings['local_compute'] = dict(mode='adam_prime_v1', ssh_host=args.md_gbsa_local_host,
        suite=args.md_gbsa_local_schrodinger, root=root, helper=root+'/md_gbsa_local_native.py',
        slots=args.md_gbsa_local_cpus, files={name:digest(Path(__file__).with_name(name)) for name in
            ('md_gbsa_native.py', 'md_gbsa_local_native.py')})
    settings['protocol'].update(prime_job_type='REAL_MIN', prime_flexdist=6.0,
                                prime_flexgroup='residue')
    return settings


def owner(settings):
    return dict(mode='adam_prime_v1', source_root=settings['root'], local=settings['local_compute'])


def control_helper(settings):
    """Versioned operational adapter; frozen scientific helpers stay untouched."""
    root = Path(settings['root'])
    marker = json.loads((root/'owner.json').read_text())
    local = marker.get('local', {})
    if any(local.get(key) != settings.get(key) for key in ('root','helper','suite','ssh_host')):
        raise RuntimeError('Adam control adapter ownership mismatch')
    source = Path(__file__).with_name('md_gbsa_jobcontrol.py')
    expected = digest(source)
    target = root/'control'/('jobcontrol-'+expected+'.py')
    if not target.exists():
        atomic_copy(source, target)
    if digest(target) != expected:
        raise RuntimeError('Adam operational adapter checksum differs')
    return target


def prepare(settings):
    local = settings['local_compute']
    root = Path(local['root'])
    # A random, frozen child of the configured share is the only writable tree.
    root.mkdir(parents=True, exist_ok=True)
    marker = root/'owner.json'
    if marker.exists():
        if json.loads(marker.read_text()) != owner(settings):
            raise RuntimeError('Shared Adam directory belongs to another request')
    else:
        if any(root.iterdir()):
            raise RuntimeError('Unowned shared Adam directory; retained')
        save(marker, owner(settings))
    for name, expected in local['files'].items():
        target = root/name
        if target.exists():
            if digest(target) != expected:
                raise RuntimeError('Frozen Adam helper changed')
        else:
            source = Path(__file__).with_name(name)
            if digest(source) != expected:
                raise RuntimeError('Cannot recover missing frozen Adam helper with changed source')
            atomic_copy(source, target)
    result = md_gbsa.response(md_gbsa.remote_call(local, 'check', str(root)))
    if result.get('owner') != owner(settings):
        raise RuntimeError('Adam does not see the configured shared directory')


def atomic_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=target.name+'.')
    try:
        with os.fdopen(fd, 'wb') as output, open(source, 'rb') as stream:
            shutil.copyfileobj(stream, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def download(host, source, target, expected):
    target = Path(target)
    if target.exists():
        if digest(target) != expected:
            raise RuntimeError('Frozen shared frame/input differs: '+str(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=target.name+'.')
    try:
        with os.fdopen(fd, 'wb') as output:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                host, 'cat -- '+shlex.quote(source)], stdout=output, stderr=subprocess.PIPE, timeout=240)
            output.flush()
            os.fsync(output.fileno())
        if result.returncode or digest(temporary) != expected:
            raise RuntimeError('MD frame transfer failed or checksum differs; no Adam submission')
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def task_root(settings, cid, replica):
    return Path(settings['local_compute']['root'])/f'CMPID_{cid}'/f'replica{replica+1:02d}'


def poll(settings, cid, replica, remote_root, archive, *, allow_submit=True, prepare_only=False):
    # Reverify upstream native completion on its own JobServer before using frames.
    result = md_gbsa.response(md_gbsa.remote_call(settings, 'frames', remote_root))
    if result['status'] != 'frames':
        return result
    local = settings['local_compute']
    root = task_root(settings, cid, replica)
    if not allow_submit and not (root/'prime/prime.job.json').exists():
        return dict(status='pending', phase='prime', reason='Cross-project Prime submission fence')
    protocol = dict(settings['protocol'], seed=1+(settings['protocol']['seed']+replica-1)%2147483647)
    config = dict(mode='adam_prime_v1', id=cid, slots=local['slots'], protocol=protocol,
                  files=result['files'], source_root=remote_root)
    marker = root/'input.json'
    if marker.exists() and json.loads(marker.read_text()) != config:
        raise RuntimeError('Frozen Adam request differs; no resubmission')
    required = {'prime/frames.maegz', 'frames.json', 'identity.json', 'md/md.job.json'}
    if set(config['files']) != required:
        raise RuntimeError('Unexpected MD snapshot transfer manifest')
    for name, expected in config['files'].items():
        download(settings['ssh_host'], remote_root+'/'+name, root/name, expected)
    if not marker.exists():
        save(marker, config)  # Publish only once every checksummed input exists.
    if prepare_only:
        return dict(status='pending', phase='prime', reason='Frames prepared for pooled Prime batch')
    result = md_gbsa.response(md_gbsa.remote_call(local, 'poll', str(root)))
    if result['status'] == 'done':
        if result['id'] != cid or result['protocol'] != protocol:
            raise RuntimeError('Adam result identity/protocol mismatch')
        # Collect coordinates and launch evidence before the database can commit.
        destination = Path(archive)/f'CMPID_{cid}'/f'replica{replica+1:02d}'
        for name in ('prime/prime_mmgbsa-out.maegz', 'prime/prime.job.json',
                     'prime/prime.launch.log', 'md/md.job.json', 'input.json', 'result.json', 'frames.json'):
            source = root/name
            if source.is_file():
                atomic_copy(source, destination/name)
        if not (destination/'prime/prime_mmgbsa-out.maegz').is_file():
            raise RuntimeError('Adam output coordinates not collected; inputs retained')
        save(destination/'collected.json', dict(output_sha256=digest(destination/'prime/prime_mmgbsa-out.maegz'),
                                                result=result))
    return result


def cleanup_committed(args, iteration, settings, directory):
    """Never remove failed, uncommitted, uncertain or foreign shared work."""
    local = settings['local_compute']
    base = Path(local['root'])
    pending = database.pending_md_local_cleanup_ids(args.db_path, iteration)
    if not base.exists():
        if pending:
            raise RuntimeError('Adam shared inputs disappeared before verified cleanup; reconcile collected results')
        return
    if json.loads((base/'owner.json').read_text()) != owner(settings):
        raise RuntimeError('Refusing cleanup of unowned Adam directory')
    for cid in pending:
        target = base/f'CMPID_{cid}'
        tombstone = base/f'.collected-CMPID_{cid}'
        receipt = Path(directory)/'adam-results'/f'CMPID_{cid}'/'cleanup.json'
        if target.exists():
            results = []
            for replica in range(settings['replicates']):
                archive = receipt.parent/f'replica{replica+1:02d}'
                collected = json.loads((archive/'collected.json').read_text())
                if digest(archive/'prime/prime_mmgbsa-out.maegz') != collected['output_sha256']:
                    raise RuntimeError('Collected Adam coordinates changed; shared files retained')
                verified = md_gbsa.response(md_gbsa.remote_call(local, 'verify', str(task_root(settings,cid,replica))))
                if verified != collected['result']:
                    raise RuntimeError('Adam result differs from collected result; shared files retained')
                results.append(verified)
            save(receipt, dict(owner=owner(settings), id=cid, results=results))
            target.rename(tombstone)
        if tombstone.exists():
            if json.loads(receipt.read_text())['owner'] != owner(settings):
                raise RuntimeError('Missing committed cleanup receipt')
            shutil.rmtree(tombstone)
        if receipt.exists():
            if json.loads(receipt.read_text())['owner'] != owner(settings):
                raise RuntimeError('Adam cleanup receipt belongs to another request')
            database.finish_md_local_cleanup(args.db_path, iteration, cid)
        else:
            raise RuntimeError('Missing Adam cleanup receipt; shared inputs retained')
    cleanup_pools(args, iteration, settings, directory)


def cleanup_pools(args, iteration, settings, directory):
    """Retire pooled inputs only after EVERY member has been committed/collected."""
    base=Path(settings['local_compute']['root'])
    scored=database.completed_md_ids(args.db_path,iteration)
    pending=database.pending_md_local_cleanup_ids(args.db_path,iteration)
    for request in base.glob('pools/batch-*/request.json'):
        batch=request.parent
        data=json.loads(request.read_text())
        if not (batch/'published.json').exists():
            continue
        ready=True
        for member in data['members']:
            cid=member['id']
            receipt=Path(directory)/'adam-results'/f'CMPID_{cid}'/'cleanup.json'
            if (cid not in scored or cid in pending or (base/f'CMPID_{cid}').exists()
                    or not receipt.exists() or json.loads(receipt.read_text())['owner']!=owner(settings)):
                ready=False
                break
        if not ready:
            continue
        archive=Path(directory)/'adam-results/pools'/batch.name
        for name in ('request.json','input.json','published.json','prime/prime.job.json',
                     'prime/prime.launch.log','prime/prime_mmgbsa.log'):
            if (batch/name).is_file():
                atomic_copy(batch/name,archive/name)
        save(archive/'cleanup.json',dict(owner=owner(settings),batch=batch.name,members=data['members']))
        batch.rename(base/('.collected-pool-'+batch.name))
    for tombstone in base.glob('.collected-pool-batch-*'):
        name=tombstone.name.removeprefix('.collected-pool-')
        receipt=Path(directory)/'adam-results/pools'/name/'cleanup.json'
        if not receipt.exists() or json.loads(receipt.read_text())['owner']!=owner(settings):
            raise RuntimeError('Missing pooled cleanup receipt; inputs retained')
        shutil.rmtree(tombstone)


def finish_cleanup(settings, ids):
    """Called only after unregistering a resolved run with all results collected."""
    base = Path(settings['local_compute']['root'])
    if not base.exists():
        return
    if json.loads((base/'owner.json').read_text()) != owner(settings):
        raise RuntimeError('Refusing cleanup of unowned Adam run')
    if any((base/f'CMPID_{cid}').exists() or (base/f'.collected-CMPID_{cid}').exists() for cid in ids):
        return  # Failed calculations retain their inputs, logs and checkpoints.
    shutil.rmtree(base)
