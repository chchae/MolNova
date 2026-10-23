"""Standalone Schrödinger helper; run using the installed suite's Python.

No SQLite or MolNova imports. Native jobs own the scientific calculations;
this helper journals submissions and collects only verified terminal output.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time

TERMINAL = {'DONE', 'FAILED', 'STOPPED', 'CANCELED'}
LIGAND = 'i_molnova_md_ligand'
SOLUTE = 'i_molnova_md_solute'
ATOM = 'i_molnova_md_atom'


def save(path, value):
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def export_pairs(source, destination, ids):
    from schrodinger import structure
    wanted = set(ids)
    seen = set()
    receptor = None
    for st in structure.StructureReader(str(source)):
        match = re.fullmatch(r'CMPID_(\d+)', st.title or '')
        cid = int(match[1]) if match else st.property.get('i_user_compound_id')
        if cid is None:
            if receptor is not None:
                raise ValueError('Ambiguous shared Prime receptor')
            receptor = st
            continue
        if cid not in wanted:
            continue
        if cid in seen:
            raise ValueError(f'Duplicate final Prime pose for {cid}')
        lig = [a.index for a in st.atom if a.property.get('i_molnova_batch_ligand') == 1
               or a.property.get('i_psp_Prime_MMGBSA_Ligand') == 1]
        if lig and len(lig) < st.atom_total:
            complex_st = st.copy()
        elif receptor is not None:
            # Frozen-receptor Prime PV: its published common receptor is exact.
            lig = list(range(receptor.atom_total + 1, receptor.atom_total + st.atom_total + 1))
            complex_st = receptor.merge(st, copy_props=True)
        else:
            raise ValueError(f'No complete optimized receptor/ligand pair for {cid}')
        lig = set(lig)
        if not lig or len(lig) == complex_st.atom_total:
            raise ValueError('Invalid Prime ligand identity')
        for atom in complex_st.atom:
            atom.property[LIGAND] = int(atom.index in lig)
            atom.property[SOLUTE] = 1
            atom.property[ATOM] = atom.index
        complex_st.title = f'CMPID_{cid}'
        complex_st.property['i_user_compound_id'] = cid
        folder = destination / f'CMPID_{cid}'
        folder.mkdir(parents=True, exist_ok=True)
        complex_st.write(str(folder / 'complex.maegz'))
        save(folder / 'identity.json', dict(id=cid, atoms=complex_st.atom_total,
                                          ligand_atoms=len(lig)))
        seen.add(cid)
    if seen != wanted:
        raise ValueError(f'Missing optimized Prime pairs: {sorted(wanted-seen)}')


def workflow(protocol, suite):
    """Use installed NPT relaxation stages, followed by separate production."""
    templates = list(suite.glob('mmshare-*/data/desmond/desmond_npt_relax.msj'))
    if len(templates) != 1:
        raise ValueError('Cannot identify installed Desmond NPT relaxation template')
    template = templates[0].read_text()
    index = template.index('simulate {')
    task = template[:index].replace('set_family = {', 'set_family = {\n md = { jlaunch_opt = ["-gpu"] }', 1)
    preparation = '''
build_geometry {
  solvent = SPC
  neutralize_system = true
  salt = { positive_ion = Na negative_ion = Cl concentration = 0.15 }
  box = { shape = orthorhombic size = [10.0 10.0 10.0] size_type = buffer }
  rezero_system = false
  override_forcefield = OPLS4
}
assign_forcefield { forcefield = OPLS4 }
'''
    production_ps = protocol['production_ns'] * 1000
    interval = max(.006, math.ceil(production_ps / (protocol['snapshots'] * 4) / .006) * .006)
    common = '''
  annealing = off
  temperature = 300.0
  pressure = 1.01325
  timestep = [0.002 0.002 0.006]
  ensemble = { class = NPT method = Langevin thermostat.tau = 1.0 barostat.tau = 2.0 }
  restraints.existing = ignore
  polarization_restraints = none
  trajectory.center = solute
'''
    equilibrium = '\nsimulate {\n title = "MolNova equilibration"\n time = %s\n%s\n}\n' % (
        protocol['equilibration_ns'] * 1000, common)
    production = '''
simulate {
  title = "MolNova production"
  no_concat_before = true
  jobname = "$MAINJOBNAME"
  dir = "."
  compress = ""
  time = %s
  trajectory.interval = %s
  checkpt.write_last_step = true
  randomize_velocity.seed = %s
%s
}
''' % (production_ps, interval, protocol['seed'], common)
    return task + preparation + template[index:] + equilibrium + production



def preflight(suite, gpu_host, cpu_host):
    hosts = subprocess.run([str(suite/'jsc'),'hosts','list','--verbose'],
                           check=True,capture_output=True,text=True,timeout=30).stdout
    for host in (gpu_host,cpu_host):
        if not re.search(r'^\s+Name: '+re.escape(host)+r'$',hosts,re.M):
            raise RuntimeError(f'Native host entry is missing: {host}')
    servers = re.findall(r'^\s{2}([^\s]+:\d+)\s*$',hosts,re.M)
    if not servers:
        raise RuntimeError('Cannot identify configured JobServer')
    evidence=[]
    for server in servers:
        info=subprocess.run([str(suite/'jsc'),'server-info',server],check=True,
                            capture_output=True,text=True,timeout=30).stdout
        versions={key:tuple(map(int,value.split('.'))) for key,value in
                  re.findall(r'(Server|Client) API version:\s+v([0-9.]+)',info)}
        if set(versions)!= {'Server','Client'} or versions['Client'] > versions['Server']:
            raise RuntimeError('Incompatible JobServer/client: '+info.strip()+
                               '\nConfigure an explicitly compatible md-gbsa-schrodinger or upgrade the server; no automatic downgrade.')
        evidence.append(info)
    return dict(status='ready',hosts=hosts,servers=evidence)

def native_rows(suite, job_id):
    result = subprocess.run([str(suite/'jsc'), 'info', '--json', job_id], check=True,
                            capture_output=True, text=True, timeout=30)
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if not any(row.get('jobId') == job_id for row in rows):
        raise RuntimeError('JobServer did not return the saved parent')
    return rows


def active_jobs(suite):
    result = subprocess.run([str(suite/'jsc'), 'list', '--id-only'], capture_output=True,
                            text=True, timeout=30)
    if result.returncode:
        if result.stdout.strip().startswith('No jobs matching') or result.stderr.strip().startswith('No jobs matching'):
            return []
        result.check_returncode()
    jobs = {}
    for job_id in result.stdout.splitlines():
        if job_id and job_id not in jobs:
            for row in native_rows(suite, job_id):
                if row.get('status') not in TERMINAL:
                    jobs[row['jobId']] = row
    return list(jobs.values())


def poll_job(folder, phase, command, suite):
    """Launch once, reconcile lost responses, and verify all child termination."""
    path = folder / f'{phase}.job.json'
    record = json.loads(path.read_text()) if path.exists() else None
    if record is None:
        record = dict(command=command, job_id=None, launching=True, started_at=time.time())
        save(path, record)
        with (folder/f'{phase}.launch.log').open('w') as log:
            process = subprocess.Popen(command, cwd=folder, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                found = re.search(r'JobId\s*:\s*(\S+)', line, re.I)
                if found:
                    record.update(job_id=found[1], launching=False)
                    save(path, record)
            record['client_exit'] = process.wait()
            save(path, record)
    if record['command'] != command:
        raise RuntimeError('Saved native command differs; submission retained')
    if not record.get('job_id'):
        result = subprocess.run([str(suite/'jsc'), 'list', '--any-status', '--launch-dir', str(folder), '--id-only'],
                                capture_output=True, text=True, timeout=30)
        if result.returncode and 'No jobs matching' not in (result.stdout+result.stderr):
            result.check_returncode()
        matches = set(result.stdout.split()) if not result.returncode else set()
        if len(matches) != 1:
            return dict(status='pending', reason='Uncertain launch; no automatic resubmission', native=record)
        record.update(job_id=matches.pop(), launching=False)
        save(path, record)
    rows = native_rows(suite, record['job_id'])
    if any(row.get('status') not in TERMINAL for row in rows):
        return dict(status='running', native=record, jobs=rows)
    parent = next(row for row in rows if row['jobId'] == record['job_id'])
    if parent['status'] != 'DONE':
        return dict(status='failed', error=f"Native {phase} ended {parent['status']}; checkpoint and JobId retained",
                    native=record, jobs=rows)
    if not record.get('downloaded'):
        subprocess.run([str(suite/'jsc'), 'download', '--cwd', record['job_id']], cwd=folder,
                       check=True, stdout=subprocess.DEVNULL, timeout=180)
        record['downloaded'] = True
        record['finished_at'] = time.time()
        save(path, record)
    return dict(status='done', native=record, jobs=rows)


def snapshot_indices(times, protocol):
    """Select distinct production frames; one snapshot uses the final frame."""
    duration = protocol['production_ns'] * 1000
    if not times or times[-1]-times[0] < duration * .99:
        raise ValueError('Production trajectory is incomplete')
    eligible = [i for i,t in enumerate(times) if t >= times[0]+duration*protocol['analysis_start_fraction']]
    count = protocol['snapshots']
    if len(eligible) < count:
        raise ValueError('Not enough distinct production snapshots')
    return ([eligible[-1]] if count == 1 else
            [eligible[round(i*(len(eligible)-1)/(count-1))] for i in range(count)])


def extract_frames(root, protocol):
    from schrodinger import structure
    from schrodinger.application.desmond.packages import topo, traj_util
    from schrodinger.structutils import analyze
    import numpy as np
    _, cms, trajectory = traj_util.read_cms_and_traj(str(root/'md'/'md-out.cms'))
    times = [float(frame.time) for frame in trajectory]
    indices = snapshot_indices(times,protocol)
    identity = json.loads((root/'identity.json').read_text())
    original = next(structure.StructureReader(str(root/'complex.maegz')))
    source_ligand = [a for a in original.atom if a.property.get(LIGAND) == 1]
    expected = sorted((a.element, a.formal_charge) for a in source_ligand)
    original_metals = sorted((a.element,a.formal_charge) for a in original.atom if a.atomic_number in
                             {3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,48,55,56,78,80})
    details = []
    (root/'prime').mkdir(exist_ok=True)
    with structure.StructureWriter(str(root/'prime'/'frames.maegz')) as writer:
        for n,index in enumerate(indices):
            st = topo.update_ct(cms.fsys_ct,cms,trajectory[index]).copy()
            keep = [a.index for a in st.atom if a.property.get(SOLUTE) == 1]
            if len(keep) != identity['atoms']:
                raise ValueError('Desmond changed/lost original solute atom identity')
            st = st.extract(keep)
            mapping = {a.property.get(ATOM): a for a in st.atom}
            if set(mapping) != set(range(1, original.atom_total+1)):
                raise ValueError('Original atom mapping was not retained')
            if any((mapping[a.index].element, mapping[a.index].formal_charge) != (a.element,a.formal_charge) for a in original.atom):
                raise ValueError('Original atom identity changed')
            ligand = [mapping[a.index] for a in source_ligand]
            bonds={tuple(sorted((b.atom1.property[ATOM],b.atom2.property[ATOM]))) for b in st.bond}
            original_bonds={tuple(sorted((b.atom1.index,b.atom2.index))) for b in original.bond}
            if bonds != original_bonds:
                raise ValueError('Original solute connectivity changed')
            if sorted((a.element,a.formal_charge) for a in ligand) != expected:
                raise ValueError('Trajectory ligand identity does not match Prime input')
            metals = sorted((a.element,a.formal_charge) for a in st.atom if a.atomic_number in
                             {3,4,11,12,13,19,20,21,22,23,24,25,26,27,28,29,30,31,37,38,48,55,56,78,80})
            if metals != original_metals:
                raise ValueError('Original ion/metal identities changed')
            # RMSD after alignment to the original receptor backbone.
            backbone = analyze.evaluate_asl(st, 'protein and backbone and not a.e H')
            if len(backbone) < 3:
                raise ValueError('No receptor backbone for trajectory QC')
            p=np.array([[st.atom[i].x,st.atom[i].y,st.atom[i].z] for i in backbone])
            q=np.array([[original.atom[st.atom[i].property[ATOM]].x,original.atom[st.atom[i].property[ATOM]].y,original.atom[st.atom[i].property[ATOM]].z] for i in backbone])
            pc,qc=p.mean(0),q.mean(0)
            u,_,vt=np.linalg.svd((p-pc).T@(q-qc))
            rotation=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
            source_heavy=np.array([[a.x,a.y,a.z] for a in source_ligand if a.atomic_number>1])
            heavy=np.array([[a.x,a.y,a.z] for a in ligand if a.atomic_number>1])
            rmsd=float(np.sqrt(np.mean(np.sum(((heavy-pc)@rotation+qc-source_heavy)**2,axis=1))))
            st.title=f"CMPID_{identity['id']}_FRAME_{n:04d}"
            st.property['i_user_compound_id']=identity['id']
            st.property['i_molnova_frame']=n
            st.property['r_molnova_frame_time_ps']=times[index]
            st.property.pop('s_chorus_trajectory_file',None)
            writer.append(st)
            details.append(dict(frame=n,time_ps=times[index],ligand_rmsd=rmsd))
    save(root/'frames.json', details)


def collect_scores(root, protocol):
    from schrodinger import structure
    identity=json.loads((root/'identity.json').read_text())
    values={}
    for st in structure.StructureReader(str(root/'prime'/'prime_mmgbsa-out.maegz')):
        match=re.fullmatch(rf"CMPID_{identity['id']}_FRAME_(\d+)", st.title or '')
        if match is None:
            raise ValueError('Unrecognized snapshot output identity')
        index=int(match[1])
        value=st.property.get('r_psp_MMGBSA_dG_Bind')
        if index in values or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Duplicate or nonfinite snapshot energy')
        values[index]=float(value)
    if set(values)!=set(range(protocol['snapshots'])):
        raise ValueError('Missing or unexpected snapshot energies; mean not published')
    frames=json.loads((root/'frames.json').read_text())
    for frame in frames:
        frame['score']=values[frame['frame']]
    journals={phase:json.loads((root/phase/f'{phase}.job.json').read_text()) for phase in ('md','prime')}
    result=dict(status='done', id=identity['id'], mean=statistics.mean(values.values()),
                std=statistics.stdev(values.values()) if len(values)>1 else 0.0,
                snapshots=frames, protocol=protocol, native=journals)
    save(root/'result.json',result)
    return result


def poll(root, suite):
    config=json.loads((root/'input.json').read_text())
    protocol=config['protocol']
    if (root/'result.json').exists():
        # A durable result is written only after both native parents/children
        # terminated and every frame was validated. Reverify journals on retry.
        for phase in ('md','prime'):
            record=json.loads((root/phase/f'{phase}.job.json').read_text())
            rows=native_rows(suite,record['job_id'])
            if next(r for r in rows if r['jobId']==record['job_id'])['status'] != 'DONE':
                raise RuntimeError('Cached result has no successful native parent')
            if any(r.get('status') not in TERMINAL for r in rows):
                raise RuntimeError('Native job unexpectedly remains active')
        return json.loads((root/'result.json').read_text())
    md=root/'md'
    md.mkdir(exist_ok=True)
    msj=workflow(protocol,suite)
    target=md/'md.msj'
    if target.exists() and target.read_text()!=msj:
        raise ValueError('Installed relaxation protocol changed; existing run retained')
    if not target.exists():
        target.write_text(msj)
    command=[str(suite/'utilities/multisim'), '-HOST', config['gpu_host']+':1', '-maxjob','1',
             '-JOBNAME','md','-m','md.msj','-mode','umbrella','-o','md-out.cms',str(root/'complex.maegz')]
    state=poll_job(md,'md',command,suite)
    if state['status']!='done':
        return dict(phase='md',**state)
    if not (root/'frames.json').exists():
        try:
            extract_frames(root,protocol)
        except ValueError as exc:
            return dict(status='failed',phase='analysis',error=str(exc))
    prime=root/'prime'
    command=[str(suite/'prime_mmgbsa'), 'frames.maegz','-JOBNAME','prime_mmgbsa',
             '-HOST',config['cpu_host']+':'+str(config.get('prime_slots',7)),'-NJOBS',str(protocol['snapshots']),
             '-ligand','atom.'+LIGAND+' 1','-job_type','ENERGY','-frozen',
             '-out_type','COMPLEX','-report_prime_log','yes','-OVERWRITE']
    # A single serialized native parent reserves seven slots. Native JobDJ
    # handles frame batching; no one-CPU job is submitted for every frame here.
    lock_path=Path(tempfile.gettempdir())/f'molnova-schrodinger-submit-{os.getuid()}.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not (prime/'prime.job.json').exists():
            jobs=active_jobs(suite)
            if any('glide' in (j.get('commandLine','')+' '+j.get('jobName','')).lower()
                   or 'mmgbsa' in (j.get('commandLine','')+' '+j.get('jobName','')).lower() for j in jobs):
                return dict(status='pending',phase='prime',reason='Glide/Prime submission fence')
        state=poll_job(prime,'prime',command,suite)
    if state['status']!='done':
        return dict(phase='prime',**state)
    try:
        return collect_scores(root,protocol)
    except ValueError as exc:
        return dict(status="failed",phase="validation",error=str(exc))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['export','poll','jobs','check'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--ids')
    args=parser.parse_args()
    suite=Path(os.environ['SCHRODINGER'])
    if args.action=='export':
        export_pairs(args.source,args.root,json.loads(args.ids))
        return
    if args.action=='check':
        config=json.loads(args.ids)
        print('MOLNOVA_MD='+json.dumps(preflight(suite,config['gpu_host'],config['cpu_host'])))
        return
    if args.action=='jobs':
        jobs=active_jobs(suite)
        for path in args.root.glob('CMPID_*/replica*/prime/prime.job.json'):
            record=json.loads(path.read_text())
            if not record.get('job_id'):
                jobs.append(dict(jobId='uncertain:'+str(path),status='UNCERTAIN',jobName='prime_mmgbsa'))
        print('MOLNOVA_MD='+json.dumps(jobs))
        return
    args.root=args.root.resolve()
    with (args.root/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=poll(args.root,suite)
    print('MOLNOVA_MD='+json.dumps(result))


if __name__=='__main__':
    main()
