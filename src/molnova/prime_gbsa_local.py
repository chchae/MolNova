"""SQLite-owning worker adapter for shared-folder Adam Prime batches."""
import json
from pathlib import Path
import shutil
import time
import uuid

from molnova import database, md_gbsa, prime_gbsa_async as prime
from molnova.md_gbsa_local import atomic_copy, digest
from molnova.md_gbsa_native import save
from molnova.schrodinger_retry import license_unavailable

MODE='adam_prime_batch_v1'


def create_batch(args,directory,compound_ids):
    from molnova.config import prime_gbsa_flexdist
    from molnova.prime_gbsa_batch import other_submission,prepare_complexes
    from molnova.schrodinger_guard import SubmissionBlocked
    directory=Path(directory)
    if prime.load_job(directory) is not None:raise RuntimeError('Saved Prime submission requires recovery')
    if other_submission(args,directory):raise SubmissionBlocked('Another iteration requires Prime recovery')
    flex=prime_gbsa_flexdist({'prime-gbsa-flexdist':getattr(args,'prime_gbsa_flexdist',0.)})
    if flex:prepare_complexes(args,directory,compound_ids)
    else:shutil.copy2(directory/'prime_gbsa_input.maegz',directory/'prime_gbsa.maegz')
    root=str(Path(args.prime_gbsa_local_work_dir)/('molnova-prime-'+uuid.uuid4().hex))
    backend=dict(root=root,ssh_host=args.prime_gbsa_local_host,suite=args.prime_gbsa_local_schrodinger,
                 helper=root+'/prime_gbsa_local_native.py')
    command=[backend['suite']+'/prime_mmgbsa','prime_gbsa.maegz','-JOBNAME','prime_gbsa',
        '-OVERWRITE','-report_prime_log','yes','-HOST','localhost:'+str(args.prime_gbsa_local_cpus),
        '-NJOBS',str(len(compound_ids))]
    if flex:command+=['-ligand','atom.i_molnova_batch_ligand 1','-rflexdist',str(flex),
                       '-rflexgroup','side','-out_type','COMPLEX']
    helpers={name:digest(Path(__file__).with_name(name)) for name in
             ('md_gbsa_native.py','md_gbsa_jobcontrol.py','prime_gbsa_local_native.py')}
    owner=dict(mode=MODE,compound_ids=list(compound_ids),command=command,slots=args.prime_gbsa_local_cpus,
               input_sha256=digest(directory/'prime_gbsa.maegz'),helpers=helpers,backend=backend,
               prime_gbsa_flexdist=flex)
    record=dict(mode=MODE,compound_ids=list(compound_ids),backend=backend,owner=owner,command=command,
                job_id=None,launching=False,started_at=time.time(),attempt=0,retry_at=0,
                license_retries=getattr(args,'prime_gbsa_license_retries',3),
                license_retry_seconds=getattr(args,'prime_gbsa_license_retry_seconds',300))
    save(prime.job_file(directory),record)
    return record


def prepare(directory,record):
    root=Path(record['backend']['root'])
    if not root.exists() and (record.get('prepared') or record.get('job_id') or record.get('launching')):
        raise RuntimeError('Saved Adam Prime folder is missing; no recreation/resubmission')
    root.mkdir(parents=True,exist_ok=True)
    marker=root/'owner.json'
    if marker.exists():
        if json.loads(marker.read_text())!=record['owner']:raise RuntimeError('Foreign Adam Prime owner')
    else:
        if any(root.iterdir()):raise RuntimeError('Unowned Adam Prime folder; retained')
        save(marker,record['owner'])
    for name,expected in record['owner']['helpers'].items():
        target=root/name
        if not target.exists():
            source=Path(__file__).with_name(name)
            if digest(source)!=expected:raise RuntimeError('Missing frozen Adam Prime helper; source changed')
            atomic_copy(source,target)
        if digest(target)!=expected:raise RuntimeError('Frozen Adam Prime helper changed')
    target=root/'input/prime_gbsa.maegz'
    if not target.exists():
        source=Path(directory)/'prime_gbsa.maegz'
        if digest(source)!=record['owner']['input_sha256']:raise RuntimeError('Frozen local Prime input changed')
        atomic_copy(source,target)
    if digest(target)!=record['owner']['input_sha256']:raise RuntimeError('Shared Prime input changed')
    response=md_gbsa.response(md_gbsa.remote_call(record['backend'],'check',str(root)))
    if response.get('owner')!=record['owner']:raise RuntimeError('Adam cannot see the same shared Prime input')
    record['prepared']=True
    save(prime.job_file(directory),record)


def collect(directory,record):
    root=Path(record['backend']['root'])
    destination=Path(directory)
    for path in (root/'prime').iterdir():
        if path.is_file() and (path.suffix=='.log' or path.name=='prime.job.json' or path.name=='prime_gbsa-out.maegz'):
            atomic_copy(path,destination/path.name)


def stream(directory,args,record,state):
    from molnova import _core as core
    import re
    done=set(record.get('imported_subjobs',[]))
    for child in state.get('subjobs',[]):
        identity=child['job_id']
        if identity in done:continue
        parent=record['job_id']
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]+',value) or value in ('.','..') for value in (identity,parent)):
            raise RuntimeError('Invalid Adam child output identity')
        source=Path(record['backend']['root'])/'subjobs'/identity/'result.maegz'
        if digest(source)!=child['sha256']:raise RuntimeError('Adam child output checksum differs')
        folder=Path(directory)/'subjobs'/parent/identity
        atomic_copy(source,folder/'result.maegz')
        extractor=core.create_prime_gbsa_score_extractor(folder)
        scores=folder/'scores.tsv'
        core.run_command([args.schrodinger/'run',extractor,folder/'result.maegz',scores],cwd=folder)
        updated=core.update_prime_gbsa_scores(scores,args.db_path,allowed_ids=record['compound_ids'])
        done.add(identity)
        record['imported_subjobs']=sorted(done)
        save(prime.job_file(directory),record)
        print(f'Adam Prime child {identity}: saved={updated}.',flush=True)


def cleanup(directory,record):
    from molnova import schrodinger_guard as guard
    root=Path(record['backend']['root'])
    tombstone=root.with_name(root.name+'.collected')
    receipt=Path(directory)/('adam-cleanup-'+root.name+'.json')
    if root.exists():
        if json.loads((root/'owner.json').read_text())!=record['owner']:raise RuntimeError('Foreign Adam Prime cleanup')
        state=md_gbsa.response(md_gbsa.remote_call(record['backend'],'verify',str(root)))
        if state['output_sha256']!=digest(Path(directory)/'prime_gbsa-out.maegz'):
            raise RuntimeError('Collected Prime coordinates differ; shared work retained')
        save(receipt,dict(owner=record['owner'],native=state.get('native'),output_sha256=state['output_sha256']))
        guard.unregister_md_remote(record['backend'])
        root.rename(tombstone)
    if tombstone.exists():
        if not receipt.exists() or json.loads(receipt.read_text())['owner']!=record['owner']:
            raise RuntimeError('Missing Adam Prime cleanup receipt')
        shutil.rmtree(tombstone)
    if not receipt.exists():raise RuntimeError('Missing Adam Prime collection/cleanup evidence')
    guard.unregister_md_remote(record['backend'])


def poll_batch(iteration,args,directory,record):
    from molnova import _core as core, schrodinger_guard as guard
    from molnova.prime_gbsa_batch import publish_coordinates
    directory=Path(directory)
    root=Path(record['backend']['root'])
    if record.get('collected') and (not root.exists() or root.with_name(root.name+'.collected').exists()):
        cleanup(directory,record)
        return retire(directory,record,0)
    prepare(directory,record)
    guard.register_md_remote(record['backend'])
    if record.get('retry_at',0)>time.time():
        print('Adam Prime: waiting for saved license retry time.',flush=True)
        return None
    try:
        with guard.submission_guard(args.schrodinger,'prime_gbsa',md_controller=True):
            if record.get('recovery_token'):
                md_gbsa.response(md_gbsa.remote_call(record['backend'],'retry',str(root),'--ids',record['recovery_token']))
                record.pop('recovery_token')
                record.update(job_id=None,launching=False)
                save(prime.job_file(directory),record)
            if not (root/'prime/prime.job.json').exists() and any(
                    guard.job_kind(j)=='prime_gbsa' for j in guard.active_jobs(args.schrodinger)):
                return None
            state=md_gbsa.response(md_gbsa.remote_call(record['backend'],'poll',str(root)))
    except guard.SubmissionBlocked as exc:
        print(str(exc),flush=True)
        return None
    if state.get('native'):
        record['job_id']=state['native'].get('job_id')
        record['launching']=state['native'].get('launching',False)
        record['native']=state['native']
        save(prime.job_file(directory),record)
    stream(directory,args,record,state)
    if state['status']=='failed':
        collect(directory,record)
        if record.get('imported_subjobs'):publish_coordinates(args,directory,record['compound_ids'])
        if license_unavailable(state.get('diagnostics','')) and record['attempt']<record['license_retries']:
            record.update(attempt=record['attempt']+1,retry_at=time.time()+record['license_retry_seconds'],
                          recovery_token=uuid.uuid4().hex)
            save(prime.job_file(directory),record)
            return None
        raise prime.PrimeReconciliationPending('Adam Prime native batch failed; inputs, JobId and results retained')
    if state['status']!='done':
        print(f'Iteration {iteration}: Adam Prime {state["status"]}; slots={record["owner"]["slots"]}.',flush=True)
        return None
    collect(directory,record)
    if digest(directory/'prime_gbsa-out.maegz')!=state['output_sha256']:
        raise RuntimeError('Adam Prime output collection checksum differs')
    diagnostics='\n'.join(p.read_text(errors='replace') for p in directory.glob('*.log'))
    failed=sorted(set(prime.atomtyping_failed_ids(diagnostics,record['compound_ids'])) & set(state['missing_ids']))
    if set(state['missing_ids'])-set(failed):
        raise prime.PrimeReconciliationPending('Missing Adam Prime compound results without confirmed atomtyping failures')
    updated=core.finish_iteration_prime_gbsa(iteration,args,directory,record['compound_ids'])
    if failed:database.mark_prime_gbsa_atomtyping_failures(args.db_path,failed,str(prime.PrimeAtomTypingError(failed)))
    publish_coordinates(args,directory,record['compound_ids'])
    scored=database.prime_gbsa_scored_ids(args.db_path,iteration)
    if set(record['compound_ids'])-set(scored)-set(failed):
        raise prime.PrimeReconciliationPending('Adam scores not fully committed; shared inputs retained')
    record['collected']=True
    record['atomtyping_failed_ids']=failed
    save(prime.job_file(directory),record)
    if not failed:cleanup(directory,record)
    else:guard.unregister_md_remote(record['backend'])  # Preserve failed scientific inputs for recovery.
    return retire(directory,record,updated)


def retire(directory,record,updated):
    record['finished_at']=time.time()
    save(prime.job_file(directory),record)
    prime.job_file(directory).replace(Path(directory)/f'adam-batch-{time.time_ns()}.completed.json')
    print('Adam Prime batch: results collected and committed.',flush=True)
    return updated
