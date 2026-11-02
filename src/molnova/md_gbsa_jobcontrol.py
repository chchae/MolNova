"""Operational JobControl adapter around a frozen Adam scientific helper.

Only submission journals, status and file-collection semantics are adapted.
The saved Prime command, input, helper files and scientific protocol stay exact.
"""
import importlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid

LEGACY = r'[A-Za-z0-9_.-]+-[0-9]+-[0-9a-fA-F]{8}'


def is_legacy(job_id):
    return bool(re.fullmatch(LEGACY, job_id))


def legacy_rows(job_id):
    from schrodinger.job import jobcontrol
    rows = []
    seen = set()

    def visit(identity, parent=None):
        if identity in seen:
            raise RuntimeError('Repeated legacy JobControl identity; reconciliation required')
        seen.add(identity)
        job = jobcontrol.Job(identity)
        if job.JobId != identity or (parent and job.ParentJobId not in (None, '', parent)):
            raise RuntimeError('Legacy JobControl parent/child identity mismatch')
        complete = job.isComplete() and job.Status == 'completed'
        status = ('DONE' if job.ExitStatus == 'finished' else 'FAILED') if complete else 'RUNNING'
        rows.append(dict(jobId=identity, status=status, jobName=job.Name,
            commandLine=job.Commandline, parentJobId=parent or '', jobHost=job.JobHost,
            legacyStatus=job.Status, allFilesDownloaded=bool(complete and job.isDownloaded()),
            spec=dict(launchParams=dict(launchDirectory=job.Dir))))
        for child in job.SubJobs:
            visit(child, identity)
    visit(job_id)
    return rows


def legacy_ids(suite, query):
    result = subprocess.run([str(suite/'jobcontrol'), '-list', '-children', query],
                            capture_output=True, text=True, check=True, timeout=30)
    output = result.stdout.strip()
    ids = set(re.findall(r'(?<![A-Za-z0-9_.-])('+LEGACY+r')(?![A-Za-z0-9_.-])', output))
    empty = ('** no active jobs were found **', '** no matching jobs were found **',
             f"** no jobs were found that match '{query}' **")
    if not ids and output and output not in empty:
        raise RuntimeError('Unrecognized legacy JobControl listing: '+output[-2000:])
    return ids


def recover_legacy(suite, folder, record):
    from schrodinger.job import jobcontrol
    matches = []
    for identity in legacy_ids(suite, 'Dir='+str(folder)):
        job = jobcontrol.Job(identity)
        if job.ParentJobId or Path(job.Dir).resolve() != folder.resolve():
            continue
        if shlex.split(job.Commandline) != record['command']:
            continue
        launched = time.mktime(time.strptime(job.LaunchTime, '%Y-%m-%d-%H:%M:%S'))
        if launched >= record['started_at']-2:
            matches.append(identity)
    if len(matches) > 1:
        raise RuntimeError('Ambiguous legacy launch; no automatic resubmission')
    return matches[0] if matches else None


def install(native):
    original_rows = native.native_rows
    original_active = native.active_jobs
    original_poll = native.poll_job

    def rows(suite, identity):
        return legacy_rows(identity) if is_legacy(identity) else original_rows(suite, identity)

    def active(suite):
        jobs = {job['jobId']:job for job in original_active(suite)}
        for identity in legacy_ids(suite, 'active'):
            if identity in jobs:
                continue
            for job in legacy_rows(identity):
                if job['status'] not in native.TERMINAL:
                    jobs[job['jobId']] = job
        return list(jobs.values())

    def poll(folder, phase, command, suite):
        command = [str(value) for value in command]
        path = folder/f'{phase}.job.json'
        record = json.loads(path.read_text()) if path.exists() else None
        if record is not None and record['command'] != command:
            raise RuntimeError('Saved native command differs; submission retained')
        if record is None:
            record = dict(command=command, job_id=None, launching=True, started_at=time.time())
            native.save(path, record)
            with (folder/f'{phase}.launch.log').open('w') as log:
                process = subprocess.Popen(command, cwd=folder, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    match = re.search(r'JobId\s*:\s*(\S+)', line, re.I)
                    if match:
                        record.update(job_id=match[1], launching=False)
                        native.save(path, record)
                record['client_exit'] = process.wait()
                native.save(path, record)
        if not record.get('job_id'):
            identity = recover_legacy(suite, folder, record)
            if identity:
                record.update(job_id=identity, launching=False)
                native.save(path, record)
            else:
                # Existing native recovery queries its JobServer; it never
                # launches again when an intent record already exists.
                return original_poll(folder, phase, command, suite)
        if not is_legacy(record['job_id']):
            return original_poll(folder, phase, command, suite)
        jobs = legacy_rows(record['job_id'])
        parent = jobs[0]
        if (Path(parent['spec']['launchParams']['launchDirectory']).resolve() != folder.resolve()
                or shlex.split(parent['commandLine']) != command):
            raise RuntimeError('Saved legacy launch directory/command mismatch')
        if any(job['status'] not in native.TERMINAL for job in jobs):
            return dict(status='running', native=record, jobs=jobs)
        if parent['status'] != 'DONE':
            return dict(status='failed', error='Legacy Prime job failed; journal retained', native=record, jobs=jobs)
        if not parent['allFilesDownloaded']:
            return dict(status='pending', reason='Legacy output transfer pending', native=record, jobs=jobs)
        # In the installed API legacy completion includes output transfer;
        # Job.download() is explicitly a no-op. Never invoke jsc for this ID.
        if not record.get('downloaded'):
            record.update(downloaded=True, finished_at=time.time(), job_control='legacy', legacy_evidence=jobs)
            native.save(path, record)
        return dict(status='done', native=record, jobs=jobs)

    native.native_rows = rows
    native.active_jobs = active
    native.poll_job = poll


def pool_command(suite, slots, snapshots):
    return [str(suite/'prime_mmgbsa'), 'frames.maegz', '-JOBNAME', 'prime_mmgbsa',
        '-HOST', 'localhost:'+str(slots), '-NJOBS', str(snapshots),
        '-ligand', 'atom.i_molnova_md_ligand 1', '-job_type', 'REAL_MIN',
        '-rflexdist', '6.0', '-rflexgroup', 'residue', '-out_type', 'COMPLEX',
        '-report_prime_log', 'yes', '-OVERWRITE']


def pool_binding(root):
    path = root/'prime/pool.json'
    if not path.exists():
        return None
    binding = json.loads(path.read_text())
    if not re.fullmatch(r'batch-[a-f0-9]{32}', binding.get('batch','')):
        raise RuntimeError('Invalid pooled Prime batch identity')
    return root.parents[1]/'pools'/binding['batch']


def pool_members(base, helper):
    members = []
    for marker in sorted(base.glob('CMPID_*/replica*/input.json')):
        root = marker.parent
        if (root/'prime/prime.job.json').exists() or (root/'prime/pool.json').exists():
            continue
        if (root/'prime/prime.launch.log').exists() or (root/'result.json').exists():
            raise RuntimeError('Unjournaled Adam calculation artifacts; no pooled resubmission')
        config = helper.validate_input(root)
        members.append(dict(path=str(root.relative_to(base)), id=config['id'],
                            snapshots=config['protocol']['snapshots'],
                            input_sha256=hashlib.sha256(marker.read_bytes()).hexdigest()))
    return members


def validate_member(base, member, helper):
    relative = Path(member['path'])
    if (len(relative.parts)!=2 or not re.fullmatch(r'CMPID_[0-9]+',relative.parts[0])
            or not re.fullmatch(r'replica[0-9]+',relative.parts[1])):
        raise RuntimeError('Invalid pooled molecule path')
    root = base/relative
    if hashlib.sha256((root/'input.json').read_bytes()).hexdigest()!=member['input_sha256']:
        raise RuntimeError('Pooled molecule input changed')
    config = helper.validate_input(root)
    if config['id']!=member['id'] or config['protocol']['snapshots']!=member['snapshots']:
        raise RuntimeError('Pooled molecule identity/count changed')
    return root, config


def prepare_pool(batch, helper):
    """Atomic intent binds the exact molecules before any native submission."""
    from schrodinger import structure
    native = helper.native
    request = json.loads((batch/'request.json').read_text())
    base = batch.parents[1]
    target = batch/'prime/frames.maegz'
    if target.exists():
        metadata=json.loads((batch/'input.json').read_text())
        if hashlib.sha256(target.read_bytes()).hexdigest()!=metadata['frames_sha256']:
            raise RuntimeError('Frozen pooled Prime frames changed')
        return request
    members = [validate_member(base,m,helper) for m in request['members']]
    for root,config in members:
        binding = root/'prime/pool.json'
        expected = dict(batch=batch.name)
        if binding.exists() and json.loads(binding.read_text())!=expected:
            raise RuntimeError('Molecule belongs to another pooled Prime batch')
        if not binding.exists():
            native.save(binding, expected)
    if (batch/'prime/prime.job.json').exists():
        raise RuntimeError('Missing submitted pooled frames; no rewrite')
    pending=target.with_name('frames.pending.maegz')
    with structure.StructureWriter(str(pending)) as writer:
        for root,config in members:
            seen=set()
            for st in structure.StructureReader(str(root/'prime/frames.maegz')):
                match=re.fullmatch(rf"CMPID_{config['id']}_FRAME_(\d+)",st.title or '')
                if not match or int(match[1]) in seen:
                    raise RuntimeError('Invalid/duplicate pooled input snapshot identity')
                seen.add(int(match[1]))
                writer.append(st)
            if seen!=set(range(config['protocol']['snapshots'])):
                raise RuntimeError('Incomplete pooled input snapshots')
    # Metadata precedes publication, so a stop between writes can safely finish.
    native.save(batch/'input.json',dict(frames_sha256=hashlib.sha256(pending.read_bytes()).hexdigest()))
    os.replace(pending,target)
    return request


def publish_pool(batch, helper):
    """Separate complete native complexes by original compound/frame identity."""
    from schrodinger import structure
    native=helper.native
    request=json.loads((batch/'request.json').read_text())
    base=batch.parents[1]
    # Parse by compound AND frame; replicas of the same compound are assigned
    # to different batches because their native titles intentionally coincide.
    wanted={m['id']:m for m in request['members']}
    outputs={cid:[] for cid in wanted}
    for st in structure.StructureReader(str(batch/'prime/prime_mmgbsa-out.maegz')):
        match=re.fullmatch(r'CMPID_(\d+)_FRAME_(\d+)',st.title or '')
        if not match or int(match[1]) not in wanted:
            raise ValueError('Unexpected pooled output identity')
        score=st.property.get('r_psp_MMGBSA_dG_Bind')
        if not isinstance(score,(int,float)) or not math.isfinite(score):
            raise ValueError('Missing/nonfinite pooled snapshot energy')
        outputs[int(match[1])].append((int(match[2]),st))
    for cid,member in wanted.items():
        frames=outputs[cid]
        if len(frames)!=member['snapshots'] or {f[0] for f in frames}!=set(range(member['snapshots'])):
            raise ValueError('Duplicate/missing pooled output snapshots')
    record=json.loads((batch/'prime/prime.job.json').read_text())
    for member in request['members']:
        root,config=validate_member(base,member,helper)
        output=root/'prime/prime_mmgbsa-out.maegz'
        if not (root/'result.json').exists():
            pending=output.with_name('prime_mmgbsa-out.pending.maegz')
            with structure.StructureWriter(str(pending)) as writer:
                for _,st in sorted(outputs[member['id']],key=lambda item:item[0]):
                    writer.append(st)
            os.replace(pending,output)
            native.save(root/'prime/prime.job.json',dict(record,pool_batch=batch.name))
            result=native.collect_scores(root,config['protocol'])
            result['backend']='adam_prime_v1'
            native.save(root/'result.json',result)
    native.save(batch/'published.json',dict(members=request['members']))


def install_pool(helper):
    native=helper.native
    original_poll=helper.poll

    def journals(base,suite):
        jobs=[]
        seen=set()
        paths=list(base.glob('CMPID_*/replica*/prime/prime.job.json'))
        paths+=list(base.glob('pools/*/prime/prime.job.json'))
        if (base/'prime/prime.job.json').exists():
            paths.append(base/'prime/prime.job.json')
        for path in paths:
            record=json.loads(path.read_text())
            identity=record.get('job_id')
            if not identity:
                jobs.append(dict(jobId='uncertain:'+str(path),status='UNCERTAIN',jobName='prime_mmgbsa'))
            elif identity not in seen:
                seen.add(identity)
                saved_suite=Path(record['command'][0]).parent
                jobs.extend(row for row in native.native_rows(saved_suite,identity)
                            if row.get('status') not in native.TERMINAL)
        # A durable group request may precede its native journal. It still owns
        # the host reservation until that exact group can be reconciled.
        for request in base.glob('pools/*/request.json'):
            if not (request.parent/'prime/prime.job.json').exists():
                jobs.append(dict(jobId='uncertain:'+str(request),status='UNCERTAIN',jobName='prime_mmgbsa'))
        return jobs

    def poll(root,suite):
        helper.validate_input(root)
        batch=pool_binding(root)
        if batch is None and (root/'prime/prime.job.json').exists():
            return original_poll(root,suite)  # Recover exact pre-pool commands.
        lock_path=Path(tempfile.gettempdir())/f'molnova-schrodinger-submit-{os.getuid()}.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
            base=root.parents[1]
            if batch is None:
                # Recover group intent written before molecule bindings.
                for request_file in sorted(base.glob('pools/*/request.json')):
                    request=json.loads(request_file.read_text())
                    if not (request_file.parent/'prime/prime.job.json').exists():
                        prepare_pool(request_file.parent,helper)
                        batch=pool_binding(root)
                        if batch is None:
                            return dict(status='pending',phase='prime',reason='Pooled launch recovery pending')
                        break
            if batch is None:
                for peer in base.parent.glob('molnova-*'):
                    if journals(peer,suite):
                        return dict(status='pending',phase='prime',reason='Adam pooled slot reservation')
                if any('glide' in (j.get('commandLine','')+' '+j.get('jobName','')).lower()
                       or 'mmgbsa' in (j.get('commandLine','')+' '+j.get('jobName','')).lower()
                       for j in native.active_jobs(suite)):
                    return dict(status='pending',phase='prime',reason='Adam Glide/Prime fence')
                # Put this molecule first, then admit distinct other originals.
                members=pool_members(base,helper)
                members.sort(key=lambda m:m['path']!=str(root.relative_to(base)))
                distinct={}
                for member in members:
                    distinct.setdefault(member['id'],member)
                members=list(distinct.values())
                if not members:
                    raise RuntimeError('No eligible pooled frames')
                batch=base/'pools'/('batch-'+uuid.uuid4().hex)
                (batch/'prime').mkdir(parents=True)
                config=helper.validate_input(root)
                request=dict(mode='pooled_frames_v1',members=members,slots=config['slots'],
                    command=pool_command(suite,config['slots'],sum(m['snapshots'] for m in members)))
                native.save(batch/'request.json',request)
                prepare_pool(batch,helper)
            request=prepare_pool(batch,helper)
            state=native.poll_job(batch/'prime','prime',request['command'],suite)
            record_path=batch/'prime/prime.job.json'
            if record_path.exists():
                record=json.loads(record_path.read_text())
                for member in request['members']:
                    member_root=base/member['path']
                    if member_root.exists() and not (member_root/'prime/prime.job.json').exists():
                        native.save(member_root/'prime/prime.job.json',dict(record,pool_batch=batch.name))
        if state['status']!='done':
            return dict(phase='prime',**state)
        with (batch/'publish.lock').open('a') as lock:
            try:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                return dict(status='pending',phase='prime',reason='Pooled output publication in progress')
            if not (batch/'published.json').exists():
                try:
                    publish_pool(batch,helper)
                except ValueError as exc:
                    return dict(status='failed',phase='validation',error=str(exc))
        return helper.verify(root,suite)

    helper.journal_jobs=journals
    helper.poll=poll


def main():
    # Adapter is stored below <owned shared run>/control/. Import the exact
    # original helper next to owner.json; never replace its pinned source.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    helper = importlib.import_module('md_gbsa_local_native')
    install(helper.native)
    install_pool(helper)
    helper.main()


if __name__ == '__main__':
    main()
