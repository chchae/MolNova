"""Standalone Adam calculator for a native multi-compound Prime batch."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

if __package__:
    from . import md_gbsa_native as native, md_gbsa_jobcontrol as control
else:
    import md_gbsa_native as native
    import md_gbsa_jobcontrol as control


def validate(root):
    config=json.loads((root/'owner.json').read_text())
    if config['mode']!='adam_prime_batch_v1' or type(config['slots']) is not int or not 1<=config['slots']<=16:
        raise RuntimeError('Invalid Adam Prime batch ownership/budget')
    if hashlib.sha256((root/'input/prime_gbsa.maegz').read_bytes()).hexdigest()!=config['input_sha256']:
        raise RuntimeError('Frozen Adam Prime input changed')
    return config


def journals(root,suite):
    paths=list(root.glob('CMPID_*/replica*/prime/prime.job.json'))
    paths+=list(root.glob('pools/*/prime/prime.job.json'))
    if (root/'prime/prime.job.json').exists():paths.append(root/'prime/prime.job.json')
    rows=[];seen=set()
    for path in paths:
        record=json.loads(path.read_text())
        identity=record.get('job_id')
        if not identity:
            rows.append(dict(jobId='uncertain:'+str(path),status='UNCERTAIN',jobName='prime_gbsa'))
        elif identity not in seen:
            seen.add(identity)
            rows.extend(j for j in native.native_rows(Path(record['command'][0]).parent,identity)
                        if j.get('status') not in native.TERMINAL)
    for path in root.glob('pools/*/request.json'):
        if not (path.parent/'prime/prime.job.json').exists():
            rows.append(dict(jobId='uncertain:'+str(path),status='UNCERTAIN',jobName='prime_mmgbsa'))
    return rows


def read_scores(path,config):
    from schrodinger import structure
    scores={}
    for st in structure.StructureReader(str(path)):
        match=re.fullmatch(r'CMPID_(\d+)',st.title or '')
        if not match:
            # Frozen-receptor PV output includes its unscored common receptor.
            if st.property.get('r_psp_MMGBSA_dG_Bind') is not None:
                raise ValueError('Scored Adam output lacks compound identity')
            continue
        cid=int(match[1])
        if cid not in config['compound_ids'] or cid in scores:
            raise ValueError('Unexpected or duplicate Adam compound output')
        value=st.property.get('r_psp_MMGBSA_dG_Bind',st.property.get('r_psp_MMGBSA_dG_Bind(NS)'))
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Missing/nonfinite Adam Prime score')
        scores[cid]=float(value)
    return scores


def subjobs(root,config,state,suite):
    """Freeze only verified completed child output while its parent runs."""
    parent=state.get('native',{}).get('job_id')
    completed=[]
    for child in state.get('jobs',[]):
        if child.get('parentJobId')!=parent or child.get('status')!='DONE':continue
        identity=child['jobId'];name=child.get('jobName','')
        if not re.fullmatch(r'prime_gbsa\.\d+',name):continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',identity) or identity in ('.','..'):
            raise RuntimeError('Invalid Prime child identity')
        directory=root/'subjobs'/identity
        directory.mkdir(parents=True,exist_ok=True)
        target=directory/'result.maegz'
        if not target.exists():
            temporary=directory/'result.pending.maegz'
            if control.is_legacy(identity):
                from schrodinger.job import jobcontrol
                job=jobcontrol.Job(identity)
                if not job.isComplete() or job.ExitStatus!='finished' or not job.isDownloaded():continue
                source=Path(job.Dir)/(name+'-out.maegz')
                if not source.is_file():continue
                shutil.copy2(source,temporary)
            else:
                for candidate in (parent,identity):
                    with temporary.open('wb') as stream:
                        fetched=subprocess.run([str(suite/'jsc'),'tail-file','--name',name+'-out.maegz',
                            '--force',candidate],stdout=stream,stderr=subprocess.PIPE,timeout=30)
                    if fetched.returncode==0 and temporary.stat().st_size:break
                else:
                    temporary.unlink(missing_ok=True)
                    continue
            read_scores(temporary,config)
            temporary.replace(target)
        read_scores(target,config)
        completed.append(dict(job_id=identity,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    return completed


def poll(root,suite):
    config=validate(root)
    prime=root/'prime'
    prime.mkdir(exist_ok=True)
    target=prime/'prime_gbsa.maegz'
    if not target.exists():
        shutil.copy2(root/'input/prime_gbsa.maegz',target)
    elif hashlib.sha256(target.read_bytes()).hexdigest()!=config['input_sha256']:
        raise RuntimeError('Adam working input changed')
    lockpath=Path(tempfile.gettempdir())/f'molnova-schrodinger-submit-{os.getuid()}.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        intent=root/'launch-intent.json'
        if not (prime/'prime.job.json').exists() and intent.exists():
            saved=json.loads(intent.read_text())
            native.save(prime/'prime.job.json',dict(command=config['command'],job_id=None,
                launching=True,started_at=saved['started_at']))
        if not (prime/'prime.job.json').exists():
            if any(prime.glob('*.log')) or any(prime.glob('*-out.maegz')):
                raise RuntimeError('Unjournaled Adam Prime outputs; no resubmission')
            for peer in root.parent.glob('molnova-*'):
                if peer!=root and journals(peer,suite):
                    return dict(status='pending',reason='Adam CPU/license reservation')
            if any('glide' in (j.get('commandLine','')+' '+j.get('jobName','')).lower()
                   or 'mmgbsa' in (j.get('commandLine','')+' '+j.get('jobName','')).lower()
                   or 'prime_gbsa' in (j.get('commandLine','')+' '+j.get('jobName','')).lower()
                   for j in native.active_jobs(suite)):
                return dict(status='pending',reason='Adam Glide/Prime submission fence')
            native.save(intent,dict(started_at=time.time(),command=config['command']))
        state=native.poll_job(prime,'prime',config['command'],suite)
    diagnostics='\n'.join(p.read_text(errors='replace') for p in prime.glob('*.log'))
    result=dict(state,diagnostics=diagnostics[-100000:])
    result['subjobs']=subjobs(root,config,state,suite)
    if state['status']=='done':
        scores=read_scores(root/'prime/prime_gbsa-out.maegz',config)
        result.update(scores=scores,missing_ids=sorted(set(config['compound_ids'])-set(scores)),
                      output_sha256=hashlib.sha256((prime/'prime_gbsa-out.maegz').read_bytes()).hexdigest())
    return result


def retry(root,suite,token):
    config=validate(root)
    if not re.fullmatch(r'[a-f0-9]{32}',token):raise RuntimeError('Invalid license retry token')
    history=root/'license-retries'/token
    if (history/'ready.json').exists():return dict(status='ready')
    if not (history/'intent.json').exists():
        record=json.loads((root/'prime/prime.job.json').read_text())
        rows=native.native_rows(suite,record['job_id'])
        parent=next(j for j in rows if j['jobId']==record['job_id'])
        if parent['status']=='DONE' or any(j.get('status') not in native.TERMINAL for j in rows):
            raise RuntimeError('Adam license retry has live/uncertain/successful work; retained')
        history.mkdir(parents=True)
        native.save(history/'intent.json',dict(record=record,jobs=rows))
    if not (history/'prime').exists():(root/'prime').rename(history/'prime')
    if (root/'launch-intent.json').exists() and not (history/'launch-intent.json').exists():
        (root/'launch-intent.json').rename(history/'launch-intent.json')
    native.save(history/'ready.json',dict(status='ready'))
    return dict(status='ready')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['check','poll','verify','jobs','retry'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--ids')
    args=parser.parse_args();root=args.root.resolve()
    suite=Path(os.environ['SCHRODINGER'])
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    control.install(native)
    if args.action=='jobs':
        result=native.active_jobs(suite)+journals(root,suite)
    elif args.action=='check':
        native.preflight(suite,'localhost','localhost')
        result=dict(status='ready',owner=validate(root))
    else:
        with (root/'controller.lock').open('a') as lock:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.action=='retry':result=retry(root,suite,args.ids)
            else:
                if args.action=='verify' and not (root/'prime/prime.job.json').exists():
                    raise RuntimeError('Cannot verify absent native submission')
                result=poll(root,suite)
                if args.action=='verify' and result['status']!='done':raise RuntimeError('Native Prime not complete')
    print('MOLNOVA_MD='+json.dumps(result))


if __name__=='__main__':main()
