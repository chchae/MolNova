"""Cross-molecule native batching; fake structures do not validate energies."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from molnova import md_gbsa_jobcontrol as adapter, md_gbsa_native as native


@pytest.fixture
def pool(tmp_path,monkeypatch):
    base=tmp_path/'share/molnova-test'
    roots=[]
    for cid in (1,2):
        root=base/f'CMPID_{cid}/replica01'
        (root/'prime').mkdir(parents=True)
        (root/'md').mkdir()
        config=dict(id=cid,slots=16,protocol=dict(snapshots=2,seed=1,prime_job_type='REAL_MIN',
                                              prime_flexdist=6.,prime_flexgroup='residue'))
        native.save(root/'input.json',config)
        native.save(root/'identity.json',dict(id=cid))
        native.save(root/'frames.json',[dict(frame=i,time_ps=i,ligand_rmsd=.2) for i in range(2)])
        native.save(root/'md/md.job.json',dict(job_id='md-'+str(cid)))
        native.save(root/'prime/frames.maegz',[
            dict(title=f'CMPID_{cid}_FRAME_{i:04d}',property={'r_psp_MMGBSA_dG_Bind':-float(cid*10+i)})
            for i in range(2)])
        roots.append(root)
    class Writer:
        def __init__(self,path):self.path=Path(path);self.entries=[]
        def __enter__(self):return self
        def __exit__(self,*args):native.save(self.path,self.entries)
        def append(self,st):self.entries.append(dict(title=st.title,property=st.property))
    structure=SimpleNamespace(StructureWriter=Writer,
        StructureReader=lambda path:iter(SimpleNamespace(**st) for st in json.loads(Path(path).read_text())))
    monkeypatch.setitem(sys.modules,'schrodinger',SimpleNamespace(structure=structure))
    state=dict(done=False,commands=[],queries=[],legacy_calls=0)
    def rows(suite,identity):
        state['queries'].append(identity)
        return [dict(jobId=identity,status='DONE' if state['done'] else 'RUNNING',jobName='prime_mmgbsa')]
    def poll_job(folder,phase,command,suite):
        path=folder/'prime.job.json'
        if not path.exists():
            state['commands'].append(command)
            native.save(path,dict(command=command,job_id='native-parent',started_at=1))
        if state['done']:
            record=json.loads(path.read_text())
            record.update(downloaded=True,finished_at=2)
            native.save(path,record)
        return dict(status='done' if state['done'] else 'running')
    calculator=SimpleNamespace(save=native.save,TERMINAL=native.TERMINAL,poll_job=poll_job,
        native_rows=rows,active_jobs=lambda *a:[],collect_scores=native.collect_scores)
    def legacy_poll(root,suite):
        state['legacy_calls']+=1
        return dict(status='running',legacy=True)
    def verify(root,suite):
        record=json.loads((root/'prime/prime.job.json').read_text())
        assert all(j['status']=='DONE' for j in rows(suite,record['job_id']))
        return json.loads((root/'result.json').read_text())
    helper=SimpleNamespace(native=calculator,poll=legacy_poll,verify=verify,
        validate_input=lambda root:json.loads((root/'input.json').read_text()))
    adapter.install_pool(helper)
    return base,roots,state,helper


def test_ready_molecule_frames_share_one_native_sixteen_slot_queue(pool):
    base,roots,state,helper=pool
    assert helper.poll(roots[0],Path('/suite'))['status']=='running'
    assert len(state['commands'])==1
    command=state['commands'][0]
    assert command[command.index('-HOST')+1]=='localhost:16'
    assert command[command.index('-NJOBS')+1]=='4'
    batch=adapter.pool_binding(roots[0])
    assert adapter.pool_binding(roots[1])==batch
    entries=json.loads((batch/'prime/frames.maegz').read_text())
    assert {st['title'] for st in entries}=={f'CMPID_{cid}_FRAME_{i:04d}' for cid in (1,2) for i in range(2)}
    # A second molecule must recover the SAME native parent, never launch another.
    assert helper.poll(roots[1],Path('/suite'))['status']=='running'
    assert len(state['commands'])==1


def test_complete_pool_splits_original_identities_and_survives_collected_member_cleanup(pool):
    base,roots,state,helper=pool
    helper.poll(roots[0],Path('/suite'))
    batch=adapter.pool_binding(roots[0])
    shutil.copy2(batch/'prime/frames.maegz',batch/'prime/prime_mmgbsa-out.maegz')
    state['done']=True
    first=helper.poll(roots[0],Path('/suite'))
    assert first['id']==1 and first['mean']==-10.5
    assert (roots[1]/'result.json').exists()
    shutil.rmtree(roots[0].parent)  # Already collected and acknowledged by local worker.
    second=helper.poll(roots[1],Path('/suite'))
    assert second['id']==2 and second['mean']==-20.5
    assert len(state['commands'])==1


@pytest.mark.parametrize('corrupt',['missing','duplicate','nonfinite','foreign'])
def test_invalid_pooled_results_cannot_publish_any_molecule(pool,corrupt):
    base,roots,state,helper=pool
    helper.poll(roots[0],Path('/suite'))
    batch=adapter.pool_binding(roots[0])
    entries=json.loads((batch/'prime/frames.maegz').read_text())
    if corrupt=='missing':entries.pop()
    elif corrupt=='duplicate':entries[-1]=entries[-2]
    elif corrupt=='foreign':entries[-1]['title']='CMPID_999_FRAME_0001'
    else:entries[-1]['property']['r_psp_MMGBSA_dG_Bind']=float('nan')
    native.save(batch/'prime/prime_mmgbsa-out.maegz',entries)
    state['done']=True
    assert helper.poll(roots[0],Path('/suite'))['status']=='failed'
    assert not any((root/'result.json').exists() for root in roots)


def test_existing_single_molecule_job_keeps_exact_recovery_path(pool):
    base,roots,state,helper=pool
    native.save(roots[0]/'prime/prime.job.json',dict(command=['old-command'],job_id='old-parent'))
    assert helper.poll(roots[0],Path('/suite'))['legacy'] is True
    assert state['legacy_calls']==1
    assert not state['commands']
    assert helper.poll(roots[1],Path('/suite'))['status']=='pending'
    assert not state['commands']


def test_uncertain_pooled_parent_reserves_host_and_survives_restart(pool):
    base,roots,state,helper=pool
    helper.poll(roots[0],Path('/suite'))
    batch=adapter.pool_binding(roots[0])
    record=json.loads((batch/'prime/prime.job.json').read_text())
    record['job_id']=None
    native.save(batch/'prime/prime.job.json',record)
    jobs=helper.journal_jobs(base,Path('/suite'))
    assert any(job['status']=='UNCERTAIN' for job in jobs)
    assert len(state['commands'])==1


def test_pooled_input_checksum_change_blocks_recovery(pool):
    base,roots,state,helper=pool
    helper.poll(roots[0],Path('/suite'))
    batch=adapter.pool_binding(roots[0])
    (batch/'prime/frames.maegz').write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Frozen pooled Prime frames changed'):
        helper.poll(roots[1],Path('/suite'))


def test_interrupted_pool_bindings_finish_before_submission(pool):
    base,roots,state,helper=pool
    members=adapter.pool_members(base,helper)
    batch=base/'pools'/('batch-'+'a'*32)
    (batch/'prime').mkdir(parents=True)
    native.save(batch/'request.json',dict(mode='pooled_frames_v1',members=members,slots=16,
        command=adapter.pool_command(Path('/suite'),16,4)))
    # Crash before frames and per-molecule bindings were published.
    assert helper.poll(roots[0],Path('/suite'))['status']=='running'
    assert adapter.pool_binding(roots[1])==batch
    assert len(state['commands'])==1


def test_same_compound_replicas_are_not_mixed_in_a_batch(pool):
    base,roots,state,helper=pool
    replica=roots[0].with_name('replica02')
    shutil.copytree(roots[0],replica)
    helper.poll(roots[0],Path('/suite'))
    batch=adapter.pool_binding(roots[0])
    request=json.loads((batch/'request.json').read_text())
    assert len(request['members'])==len({m['id'] for m in request['members']})==2
    assert adapter.pool_binding(replica) is None


def test_unjournaled_old_launch_cannot_be_admitted_as_new_pooled_work(pool):
    base,roots,state,helper=pool
    (roots[1]/'prime/prime.launch.log').write_text('JobId: unknown-prior-job\n')
    with pytest.raises(RuntimeError,match='Unjournaled Adam calculation artifacts'):
        helper.poll(roots[0],Path('/suite'))
    assert not state['commands']
