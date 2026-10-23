"""Worker-owned retry preparation for the supervisor's latest-iteration intent."""
import json
from pathlib import Path
import shlex

from molnova import database
from molnova.paths import stage_directory


def recover_md(args, iteration, saved):
    from molnova import md_gbsa
    from molnova.md_gbsa_native import save
    requests = database.score_retry_requests(args.db_path,iteration,'md_gbsa')
    if not requests:
        return
    directory = stage_directory(args.output,iteration,'md_gbsa')
    status_file = directory/'progress.json'
    progress = json.loads(status_file.read_text()) if status_file.exists() else {}
    settings = saved['settings']
    script = Path(__file__).with_name('md_gbsa_retry.py').read_text()
    for cid, token in requests:
        if database.md_gbsa_rows(args.db_path,iteration).get(cid,(None,None))[1] is not None:
            database.finish_score_retry(args.db_path,iteration,'md_gbsa',cid,token)
            continue
        keys = [str((cid,r)) for r in range(settings['replicates'])]
        # Keep the old failure and replica successes in a durable local audit.
        history = directory/'retry_history'/token
        history.mkdir(parents=True,exist_ok=True)
        if not (history/'progress.json').exists():
            save(history/'progress.json',{k:progress[k] for k in keys if k in progress})
        for r, key in enumerate(keys):
            if progress.get(key,{}).get('status')=='done':
                continue
            root = settings['root']+f'/CMPID_{cid}/replica{r+1:02d}'
            command = shlex.join(['python3','-c',script,root,settings['suite'],token,
                                  progress.get(key,{}).get('phase','')])
            result = json.loads(md_gbsa.ssh(settings['ssh_host'],command))
            if result['status'] not in ('ready','recover'):
                raise RuntimeError('Unexpected MD retry response')
            progress.pop(key,None)
        save(status_file,progress)
        database.finish_score_retry(args.db_path,iteration,'md_gbsa',cid,token)
        print(f'CMPID_{cid}: missing MD-GBSA score rearmed; previous attempts retained.')


def recover_prime(args, iteration, directory):
    from molnova import prime_gbsa_async as prime
    if prime.load_job(directory) is not None:
        return  # Existing worker must retire this exact native submission first.
    requests = database.score_retry_requests(args.db_path,iteration,'prime_gbsa')
    for cid, token in requests:
        if database.completed_md_ids(args.db_path,iteration):
            database.finish_score_retry(args.db_path,iteration,'prime_gbsa',cid,token,rearm=False)
            continue
        row = database.prime_gbsa_submission_rows(args.db_path,[cid]).get(cid)
        if row is not None and row[2] is not None:
            database.finish_score_retry(args.db_path,iteration,'prime_gbsa',cid,token)
            continue
        records = []
        for path in Path(directory).rglob('*.json'):
            if 'retry_history' in path.parts:
                continue
            try:
                record = json.loads(path.read_text())
            except (ValueError,OSError):
                continue
            if not isinstance(record,dict) or cid not in record.get('compound_ids',[]) or 'command' not in record:
                continue
            if not record.get('job_id'):
                raise RuntimeError('Prime launch has no durable JobId; no automatic reset')
            jobs = prime._job_details(record,path.parent)
            if not any(j.get('jobId')==record['job_id'] for j in jobs) or any(
                    j.get('status') not in {'DONE','FAILED','CANCELED','STOPPED'} for j in jobs):
                raise RuntimeError('Prime parent/children active or uncertain; no automatic reset')
            records.append(dict(path=str(path),record=record,jobs=jobs))
        if not records:
            raise RuntimeError(f'CMPID_{cid}: missing Prime JobId evidence; reconcile before retry')
        from molnova.md_gbsa_native import save
        history = Path(directory)/'retry_history'/token
        history.mkdir(parents=True,exist_ok=True)
        save(history/'terminal-evidence.json',records)
        database.finish_score_retry(args.db_path,iteration,'prime_gbsa',cid,token)
        print(f'CMPID_{cid}: missing Prime score rearmed; successful scores retained.')
