"""Independent MD ensemble worker: durable SQLite ownership and remote journals."""
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import uuid

from molnova import database
from molnova.paths import stage_directory
from molnova.states import CompoundState as State


def ssh(host, command, *, input=None, timeout=240):
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command],
                            input=input,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'MD remote operation failed: {result.stderr[-3000:]} {result.stdout[-1000:]}')
    return result.stdout


def upload(host, source, destination):
    # Send bytes over SSH rather than interpolating remote scp paths.
    with open(source,'rb') as stream:
        result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
                                 'cat > '+shlex.quote(str(destination))+'.upload && mv '+
                                 shlex.quote(str(destination))+'.upload '+shlex.quote(str(destination))],
                                 stdin=stream,capture_output=True,timeout=180)
    if result.returncode:
        raise RuntimeError('MD input transfer failed: '+result.stderr.decode()[-1000:])


def response(text):
    lines=[line[len('MOLNOVA_MD='):] for line in text.splitlines() if line.startswith('MOLNOVA_MD=')]
    if len(lines)!=1:
        raise RuntimeError('Missing or ambiguous MD native response')
    return json.loads(lines[0])


def remote_call(settings, action, root, *extra):
    return ssh(settings['ssh_host'],shlex.join([settings['suite']+'/run',settings['helper'],action,root,*extra]))


def settings_for(args, iteration):
    source=stage_directory(args.output,iteration,'prime_gbsa')/'prime_gbsa_final.maegz'
    if not source.is_file():
        raise RuntimeError(f'Missing final optimized Prime complex file: {source}')
    helper=Path(__file__).with_name('md_gbsa_native.py')
    run=uuid.uuid4().hex
    root=args.md_gbsa_remote_work_dir.rstrip('/')+'/'+run
    return dict(ssh_host=args.md_gbsa_ssh_host,suite=args.md_gbsa_schrodinger,
                gpu_host=args.md_gbsa_gpu_host,cpu_host=args.md_gbsa_cpu_host,
                gpus=args.md_gbsa_gpus,prime_slots=min(7,args.md_gbsa_cpus),
                root=root,helper=root+'/native.py',source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
                replicates=args.md_gbsa_replicates,
                protocol={key:getattr(args,'md_gbsa_'+key) for key in
                          ('production_ns','equilibration_ns','snapshots','analysis_start_fraction','seed')})


def prepare(settings, ids):
    """Immutable remote inputs; ready marker appears only after every file exists."""
    root=settings['root']; host=settings['ssh_host']
    ready=ssh(host,'test -f '+shlex.quote(root+'/ready.json')+' && cat '+shlex.quote(root+'/ready.json')+' || true')
    if ready:
        if json.loads(ready)!=dict(settings=settings,ids=ids):
            raise RuntimeError('Remote frozen MD inputs differ from SQLite selection')
        return
    # Never overwrite a run that might already have launched.
    uncertain=ssh(host,'find '+shlex.quote(root)+" -name '*.job.json' -print 2>/dev/null || true")
    if uncertain.strip():
        raise RuntimeError('Remote native journal exists without preparation marker; reconcile manually')
    source=Path(settings['source']); helper=Path(__file__).with_name('md_gbsa_native.py')
    for path,expected in ((source,settings['source_sha256']),(helper,settings['helper_sha256'])):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Frozen MD input changed: {path}')
    ssh(host,'mkdir -p '+shlex.quote(root))
    upload(host,helper,settings['helper']); upload(host,source,root+'/prime_final.maegz')
    remote_call(settings,'export',root,'--source',root+'/prime_final.maegz','--ids',json.dumps(ids))
    for cid in ids:
        base=root+f'/CMPID_{cid}'
        for replica in range(settings['replicates']):
            folder=base+f'/replica{replica+1:02d}'
            ssh(host,'mkdir -p '+shlex.quote(folder)+' && cp '+shlex.join([base+'/complex.maegz',base+'/identity.json',folder+'/']))
            protocol=dict(settings['protocol'],seed=1+(settings['protocol']['seed']+replica-1)%2147483647)
            data=dict(protocol=protocol,gpu_host=settings['gpu_host'],cpu_host=settings['cpu_host'],prime_slots=settings['prime_slots'])
            ssh(host,'cat > '+shlex.quote(folder+'/input.json'),input=json.dumps(data))
    ssh(host,'cat > '+shlex.quote(root+'/ready.json.upload')+' && mv '+shlex.join([root+'/ready.json.upload',root+'/ready.json']),
        input=json.dumps(dict(settings=settings,ids=ids)))


def run(args, iteration):
    saved=database.md_gbsa_selection(args.db_path,iteration)
    if saved is None and not getattr(args,'md_gbsa_enabled',False):
        print('MD-GBSA disabled; no saved work to recover.')
        return
    if saved is None:
        # Retire native upstream work before freezing the final Prime ranking.
        from molnova.prime_gbsa_async import load_job
        from molnova.driver import external_completion_reason
        if load_job(stage_directory(args.output,iteration,'prime_gbsa')) is not None:
            print('MD-GBSA waiting for Prime submission recovery.'); return
        reason=external_completion_reason(args,iteration,'prime_gbsa')
        if reason:
            print('MD-GBSA waiting: '+reason); return
        proposed=settings_for(args,iteration)
        # Read-only compatibility check precedes SQLite claims and job launch.
        ssh(proposed['ssh_host'],'mkdir -p '+shlex.quote(proposed['root']))
        upload(proposed['ssh_host'],Path(__file__).with_name('md_gbsa_native.py'),proposed['helper'])
        response(remote_call(proposed,'check',proposed['root'],'--ids',json.dumps(proposed)))
        saved=database.freeze_md_gbsa_selection(args.db_path,iteration,proposed,
                    count=args.md_gbsa_input_count,ratio=args.md_gbsa_input_ratio,
                    target_count=args.target_count,prime_count=args.prime_gbsa_input_count,
                    amber_keep_count=database.amber_keep_count(args))
    settings=saved['settings']
    from molnova.score_recovery import recover_md
    recover_md(args,iteration,saved)
    if database.md_gbsa_completion_reason(args.db_path,iteration) is None:
        from molnova import schrodinger_guard as guard
        guard.unregister_md_remote(settings)
        print('MD-GBSA already complete; persisted ensemble results retained.')
        return
    directory=stage_directory(args.output,iteration,'md_gbsa')
    directory.mkdir(parents=True,exist_ok=True)
    from molnova.md_gbsa_native import save
    save(directory/'selection.json',saved)
    prepare(settings,saved['ids'])
    from molnova import schrodinger_guard as guard
    guard.register_md_remote(settings)
    rows=database.md_gbsa_rows(args.db_path,iteration)
    # Poll existing runs first; uncertain responses remain owned, never relaunched.
    status_file=directory/'progress.json'
    progress=json.loads(status_file.read_text()) if status_file.exists() else {}
    tasks=[(cid,r,settings['root']+f'/CMPID_{cid}/replica{r+1:02d}')
           for cid in saved['ids'] for r in range(settings['replicates'])
           if rows[cid][1] is None and rows[cid][2:]!=(State.FAILED,'md_gbsa')]
    tasks.sort(key=lambda t: str((t[0],t[1])) not in progress)
    active=sum(v.get('phase')=='md' and v.get('status') in ('running','pending') for v in progress.values())
    for cid,replica,root in tasks:
        key=str((cid,replica)); prior=progress.get(key)
        if prior is None and active>=settings['gpus']:
            continue
        # Local cross-project lock and registry bridge local/remote JobServers.
        # Prime phase is serialized and reserves at most seven native slots.
        try:
            with guard.submission_guard(args.schrodinger,'prime_gbsa',md_controller=True):
                local=guard.active_jobs(args.schrodinger)
                if any(guard.job_kind(j)=='prime_gbsa' and not j.get('molnova_md_remote') for j in local):
                    print('MD-GBSA waiting for other Prime work.'); return
                result=response(remote_call(settings,'poll',root))
                expected_protocol=dict(settings['protocol'],seed=1+(settings['protocol']['seed']+replica-1)%2147483647)
                if result.get('status')=='done' and (result.get('id')!=cid or result.get('protocol')!=expected_protocol):
                    raise RuntimeError('Remote MD ensemble identity/protocol does not match the frozen task')
        except guard.SubmissionBlocked as exc:
            print(str(exc)); return
        progress[key]=result
        save(status_file,progress)
        if prior is None and result.get('phase')=='md' and result['status'] in ('running','pending'):
            active+=1
        if prior and prior.get('phase')=='md' and result.get('phase')!='md':
            active=max(0,active-1)
        print(f"CMPID_{cid} replica {replica+1}: {result.get('phase','ensemble')} {result['status']}")
        if result['status']=='failed':
            # Keep the selected compound owned until every submitted replica retires.
            continue
    for cid in saved['ids']:
        if rows[cid][1] is not None:
            continue
        results=[progress.get(str((cid,r))) for r in range(settings['replicates'])]
        if all(v and v['status']=='done' for v in results):
            frames=[dict(frame,replica=r+1) for r,v in enumerate(results) for frame in v['snapshots']]
            values=[f['score'] for f in frames]
            merged=dict(id=cid,mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else 0.0,snapshots=frames,
                        protocol=settings['protocol'],replicate_means=[v['mean'] for v in results],
                        qc=dict(ligand_rmsd_mean=statistics.mean(f['ligand_rmsd'] for f in frames),
                                ligand_rmsd_max=max(f['ligand_rmsd'] for f in frames)),
                        remote_root=settings['root'],suite=settings['suite'],
                        native=[v.get('native',{}) for v in results])
            save(directory/f'CMPID_{cid}.json',merged)
            database.store_md_gbsa_result(args.db_path,iteration,cid,merged)
        elif all(v and v['status'] in ('done','failed') for v in results) and any(v['status']=='failed' for v in results):
            from molnova import _core
            _core.set_compound_state([cid],State.FAILED,failed_stage='md_gbsa',
                                     failure_message='; '.join(v.get('error','Native calculation failed') for v in results if v['status']=='failed')[:4000],db_path=args.db_path)
    reason=database.md_gbsa_completion_reason(args.db_path,iteration)
    if reason is None:
        guard.unregister_md_remote(settings)
    print(reason or 'MD-GBSA ensemble results complete.')
