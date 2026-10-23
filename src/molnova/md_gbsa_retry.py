"""Standalone journal recovery; no scientific calculations or scheduler changes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TERMINAL = {'DONE', 'FAILED', 'STOPPED', 'CANCELED'}


def native_rows(suite, job_id):
    result = subprocess.run([str(Path(suite)/'jsc'),'info','--json',job_id],
                            capture_output=True,text=True,check=True,timeout=30)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def retry_replica(root, suite, token, phase):
    root = Path(root)
    history = root/'retry_history'/token
    plan_file = history/'plan.json'
    if not plan_file.exists():
        evidence = {}
        for name in ('md','prime'):
            journal = root/name/(name+'.job.json')
            if not journal.exists():
                if (root/name).exists() and any((root/name).iterdir()):
                    # Frame preparation may create Prime inputs before launch.
                    if name != 'prime' or any((root/name).glob('*.launch.log')):
                        raise RuntimeError('Native launch has no durable journal; retained for reconciliation')
                continue
            record = json.loads(journal.read_text())
            if not record.get('job_id'):
                raise RuntimeError('Native launch has no JobId; retained for reconciliation')
            jobs = native_rows(suite,record['job_id'])
            parents = [r for r in jobs if r.get('jobId')==record['job_id']]
            if len(parents)!=1 or any(r.get('status') not in TERMINAL for r in jobs):
                raise RuntimeError('Native parent/children active or uncertain; no reset')
            evidence[name] = dict(record=record,jobs=jobs,status=parents[0]['status'])
        if 'md' not in evidence:
            raise RuntimeError('Missing MD journal; no automatic reset')
        if (root/'result.json').exists() and all(evidence.get(p,{}).get('status')=='DONE' for p in ('md','prime')):
            return {'status':'recover'}
        redo_md = evidence['md']['status']!='DONE' or phase=='analysis'
        redo_prime = evidence.get('prime',{}).get('status') not in (None,'DONE') or phase=='validation'
        paths = ['md','prime','frames.json','result.json'] if redo_md else (['prime','result.json'] if redo_prime else [])
        history.mkdir(parents=True,exist_ok=True)
        # Durable intent precedes all moves; rerunning the same token completes it.
        temporary = history/'plan.pending'
        temporary.write_text(json.dumps(dict(paths=paths,evidence=evidence)))
        os.replace(temporary,plan_file)
    plan = json.loads(plan_file.read_text())
    for name in plan['paths']:
        source, target = root/name, history/name
        if source.exists() and not target.exists():
            os.rename(source,target)
    if 'prime' in plan['paths'] and 'md' not in plan['paths']:
        frames = history/'prime/frames.maegz'
        if frames.exists():
            (root/'prime').mkdir(exist_ok=True)
            if not (root/'prime/frames.maegz').exists():
                temporary = root/'prime/frames.retry.pending'
                shutil.copy2(frames,temporary)
                os.replace(temporary,root/'prime/frames.maegz')
    return {'status':'ready','history':str(history)}


if __name__=='__main__':
    print(json.dumps(retry_replica(*sys.argv[1:])))
