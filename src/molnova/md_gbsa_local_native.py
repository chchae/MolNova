"""Standalone Adam Prime calculator. No SQLite, Slurm or workflow workers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

if __package__:
    from . import md_gbsa_native as native
else:
    import md_gbsa_native as native


def validate_input(root):
    config = json.loads((root/'input.json').read_text())
    if (config['mode'] != 'adam_prime_v1' or type(config['slots']) is not int
            or not 1 <= config['slots'] <= 16):
        raise RuntimeError('Invalid Adam Prime request')
    if set(config['files']) != {'prime/frames.maegz', 'frames.json', 'identity.json', 'md/md.job.json'}:
        raise RuntimeError('Unexpected Adam input manifest')
    if (config['protocol'].get('prime_job_type'), config['protocol'].get('prime_flexdist'),
            config['protocol'].get('prime_flexgroup')) != ('REAL_MIN', 6.0, 'residue'):
        raise RuntimeError('Unexpected Adam scientific protocol')
    for name, digest in config['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen Adam input changed: '+name)
    return config


def verify(root, suite):
    config = validate_input(root)
    record = json.loads((root/'prime/prime.job.json').read_text())
    rows = native.native_rows(suite, record['job_id'])
    if (next(r for r in rows if r['jobId'] == record['job_id'])['status'] != 'DONE'
            or any(r.get('status') not in native.TERMINAL for r in rows)):
        raise RuntimeError('Adam Prime is active, failed or uncertain; inputs retained')
    result = json.loads((root/'result.json').read_text())
    if result['protocol'] != config['protocol'] or result['id'] != config['id']:
        raise RuntimeError('Adam result identity mismatch')
    return result


def journal_jobs(root, suite):
    """A missing active-list entry never releases a saved native reservation."""
    jobs = []
    for path in root.glob('CMPID_*/replica*/prime/prime.job.json'):
        record = json.loads(path.read_text())
        if not record.get('job_id'):
            jobs.append(dict(jobId='uncertain:'+str(path), status='UNCERTAIN', jobName='prime_mmgbsa'))
        else:
            saved_suite = Path(record['command'][0]).parent if record.get('command') else suite
            jobs.extend(dict(row, jobName=row.get('jobName') or 'prime_mmgbsa') for row in
                        native.native_rows(saved_suite, record['job_id']) if row.get('status') not in native.TERMINAL)
    return jobs


def poll(root, suite):
    config = validate_input(root)
    protocol = config['protocol']
    command = [str(suite/'prime_mmgbsa'), 'frames.maegz', '-JOBNAME', 'prime_mmgbsa',
               '-HOST', 'localhost:'+str(config['slots']), '-NJOBS', str(protocol['snapshots']),
               '-ligand', 'atom.'+native.LIGAND+' 1', '-job_type', 'REAL_MIN',
               '-rflexdist', '6.0', '-rflexgroup', 'residue', '-out_type', 'COMPLEX',
               '-report_prime_log', 'yes', '-OVERWRITE']
    # Serialize native parents across projects on Adam. Each parent has at most
    # sixteen single-core Prime subjobs; live/uncertain work blocks another parent.
    path = Path(tempfile.gettempdir())/f'molnova-schrodinger-submit-{os.getuid()}.lock'
    with path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not (root/'prime/prime.job.json').exists():
            # JobServer cannot report a launch that has not registered yet.
            # Durable intent from any run on this share still reserves the host.
            for peer in root.parents[2].glob('molnova-*'):
                if journal_jobs(peer, suite):
                    return dict(status='pending', phase='prime', reason='Uncertain Adam launch')
            jobs = native.active_jobs(suite)
            if any('glide' in (j.get('commandLine','')+' '+j.get('jobName','')).lower()
                   or 'mmgbsa' in (j.get('commandLine','')+' '+j.get('jobName','')).lower()
                   for j in jobs):
                return dict(status='pending', phase='prime', reason='Adam Glide/Prime fence')
        state = native.poll_job(root/'prime', 'prime', command, suite)
    if state['status'] != 'done':
        return dict(phase='prime', **state)
    try:
        result = native.collect_scores(root, protocol)
    except ValueError as exc:
        return dict(status='failed', phase='validation', error=str(exc))
    result['backend'] = 'adam_prime_v1'
    native.save(root/'result.json', result)
    return result


def retry(root, suite, token):
    if not re.fullmatch(r'[a-f0-9]{32}', token):
        raise RuntimeError('Invalid Adam retry identity')
    history = root/'retry_history'/token
    plan = history/'plan.json'
    if not plan.exists():
        validate_input(root)
        path = root/'prime/prime.job.json'
        if not path.exists():
            if any((root/'prime').glob('*.launch.log')):
                raise RuntimeError('Adam launch has no journal; no retry')
            return dict(status='ready')
        record = json.loads(path.read_text())
        if not record.get('job_id'):
            raise RuntimeError('Adam launch has no JobId; no retry')
        rows = native.native_rows(suite, record['job_id'])
        if any(r.get('status') not in native.TERMINAL for r in rows):
            raise RuntimeError('Adam Prime is active or uncertain; no retry')
        if (root/'result.json').exists():
            verify(root, suite)
            return dict(status='recover')
        history.mkdir(parents=True, exist_ok=True)
        native.save(plan, dict(record=record, jobs=rows))
    if not (history/'prime').exists():
        (root/'prime').rename(history/'prime')
    (root/'prime').mkdir(exist_ok=True)
    frames = root/'prime/frames.maegz'
    if not frames.exists():
        temporary = frames.with_suffix('.pending')
        shutil.copy2(history/'prime/frames.maegz', temporary)
        os.replace(temporary, frames)
    return dict(status='ready')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['poll', 'jobs', 'check', 'verify', 'retry'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--ids')
    args = parser.parse_args()
    args.root = args.root.resolve()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    suite = Path(os.environ['SCHRODINGER'])
    if args.action == 'check':
        result = native.preflight(suite, 'localhost', 'localhost')
        # Prove that this is the SAME shared mount, not an unrelated remote path.
        result['owner'] = json.loads((args.root/'owner.json').read_text())
    elif args.action == 'jobs':
        result = native.active_jobs(suite)
        result.extend(journal_jobs(args.root, suite))
    else:
        with (args.root/'controller.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'retry':
                result = retry(args.root, suite, args.ids)
            else:
                result = verify(args.root, suite) if args.action == 'verify' else poll(args.root, suite)
    print('MOLNOVA_MD='+json.dumps(result))


if __name__ == '__main__':
    main()
