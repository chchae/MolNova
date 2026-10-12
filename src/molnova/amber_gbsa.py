"""Remote Amber GBSA screening with durable, project-isolated recovery."""
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
import uuid

from molnova import _core as c, database


def save(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def load_job(directory):
    path = Path(directory) / 'amber_job.json'
    return json.loads(path.read_text()) if path.exists() else None


def source_fingerprints(args, iteration):
    paths = [args.amber_receptor, args.output / f'iter{iteration}/glide/best_poses.maegz']
    result = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        result[str(path.resolve())] = digest.hexdigest()
    return result


def verify_sources(args, iteration, record):
    if record.get('source_fingerprints') and record['source_fingerprints'] != source_fingerprints(args, iteration):
        raise RuntimeError('Receptor/Glide poses changed since Amber submission; saved results and jobs retained')


def ssh(settings, command, **kwargs):
    options = ['ssh']
    if settings.get('ssh_config'):
        options += ['-F', settings['ssh_config']]
    return subprocess.run([*options, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                           settings['host'], command], check=True, **kwargs)


def environment_command(settings, body):
    quote = shlex.quote
    script = (
        f"set -e; source {quote(settings['conda_init'])}; "
        f"conda activate {quote(settings['remote_env'])}; "
        'source "$CONDA_PREFIX/amber.sh"; '
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; '
        + body
    )
    return 'bash -c ' + quote(script)


def export_inputs(args, iteration, folder, ids):
    """Schrödinger reads MAEGZ locally; remote Amber only receives PDB/MOL2."""
    script = folder / 'export_inputs.py'
    script.write_text(r'''
import json, re, sys
from pathlib import Path
from schrodinger import structure
root, receptor_file, poses_file = map(Path, sys.argv[1:4])
wanted = set(json.loads(sys.argv[4]))
receptors = list(structure.StructureReader(str(receptor_file)))
if len(receptors) != 1:
    raise ValueError('Amber receptor MAEGZ must contain exactly one prepared receptor')
receptor = receptors[0]
# Preserve prepared histidine protonation when translating HIS to Amber names.
residues = list(receptor.residue)
residue_index = {a.index: i for i, residue in enumerate(residues, 1) for a in residue.atom}
bonds = []
for bond in receptor.bond:
    a, b = bond.atom1, bond.atom2
    if a.element == b.element == 'S' and residue_index[a.index] != residue_index[b.index]:
        ia, ib = residue_index[a.index], residue_index[b.index]
        if residues[ia-1].pdbres.strip() not in ('CYS','CYX') or residues[ib-1].pdbres.strip() not in ('CYS','CYX'):
            raise ValueError('Unsupported receptor sulfur linkage')
        residues[ia-1].pdbres = residues[ib-1].pdbres = 'CYX'
        bonds.append('bond rec.%s.SG rec.%s.SG\n' % (ia, ib))
for residue in residues:
    if residue.pdbres.strip() == 'HIS':
        names = {a.pdbname.strip() for a in residue.atom}
        residue.pdbres = 'HIP' if {'HD1','HE2'} <= names else 'HID' if 'HD1' in names else 'HIE' if 'HE2' in names else 'HIS'
    if residue.pdbres.strip() == 'CYS' and sum(a.formal_charge for a in residue.atom) == -1:
        residue.pdbres = 'CYM'
receptor.write(str(root / 'receptor.pdb'))
(root / 'receptor_bonds.in').write_text(''.join(bonds))
entries = []
seen = set()
for ligand in structure.StructureReader(str(poses_file)):
    match = re.fullmatch(r'CMPID_(\d+)', ligand.title or '')
    cid = int(match.group(1)) if match else None
    if cid not in wanted:
        continue
    if cid in seen:
        raise ValueError('Multiple best Glide poses for CMPID_%s' % cid)
    seen.add(cid)
    folder = root / ('CMPID_%s' % cid)
    folder.mkdir(exist_ok=True)
    charge = sum(atom.formal_charge for atom in ligand.atom)
    for atom in ligand.atom:
        atom.pdbname = '%s%s' % (atom.element, atom.index)
        atom.pdbres = 'LIG'
        atom.resnum = 1
        atom.chain = 'L'
    ligand.write(str(folder / 'input.mol2'))
    entries.append({'id': cid, 'charge': charge})
if seen != wanted:
    raise ValueError('Missing Glide poses: %s' % sorted(wanted-seen))
(root / 'entries.json').write_text(json.dumps(entries))
'''.lstrip())
    poses = args.output / f'iter{iteration}/glide/best_poses.maegz'
    c.run_command([args.schrodinger / 'run', script, folder, args.amber_receptor,
                   poses, json.dumps(ids)], cwd=folder)
    entries = json.loads((folder / 'entries.json').read_text())
    # Unique stable MOL2 names are necessary for coordinate restoration after SQM.
    from molnova.amber_runner import mol2_sections
    for entry in entries:
        path = folder / f"CMPID_{entry['id']}/input.mol2"
        lines, atoms, _ = mol2_sections(path)
        for index, fields in atoms:
            fields[1] = fields[5].split('.')[0] + fields[0]
            fields[6:8] = ['1', 'LIG']
            lines[index] = ' '.join(fields)
        path.write_text('\n'.join(lines) + '\n')
    return entries


def prepare_remote(args, iteration, directory, record):
    folder = directory / record['run_id']
    folder.mkdir(parents=True, exist_ok=True)
    fingerprints = source_fingerprints(args, iteration)
    entries = export_inputs(args, iteration, folder, record['compound_ids'])
    if fingerprints != source_fingerprints(args, iteration):
        raise RuntimeError('Amber input structures changed during export')
    identities = database.amber_identity_rows(args.db_path, iteration, record['compound_ids'])
    config = {'entries': entries, 'cpus': args.amber_cpus, 'charge_method': args.amber_charge_method,
              'protocol': 'single_pose_ff14SB_GAFF2_igb5_mbondi2_salt0.15_no_entropy',
              'identities': identities}
    save(folder / 'input.json', config)
    worker = Path(__file__).with_name('amber_runner.py')
    (folder / 'worker.py').write_bytes(worker.read_bytes())
    archive = folder / 'inputs.tar'
    with tarfile.open(archive, 'w') as stream:
        for path in [folder / 'receptor.pdb', folder / 'receptor_bonds.in', folder / 'input.json', folder / 'worker.py',
                     *(folder / f"CMPID_{e['id']}/input.mol2" for e in entries)]:
            stream.add(path, arcname=str(path.relative_to(folder)))
    remote = shlex.quote(record['remote_dir'])
    settings = record['settings']
    ssh(settings, f'mkdir -p -- {remote}')
    with archive.open('rb') as stream:
        ssh(settings, f'tar -xf - -C {remote}', stdin=stream)
    record['input_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    record['source_fingerprints'] = fingerprints
    record['prepared'] = True
    save(directory / 'amber_job.json', record)


def launch(record):
    remote = shlex.quote(record['remote_dir'])
    command = f'cd {remote}; nohup python worker.py run {remote} > launcher.log 2>&1 < /dev/null &'
    # Repeated launch requests use the same directory and inherited remote lock.
    ssh(record['settings'], environment_command(record['settings'], command),
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def poll(args, iteration, directory, record):
    if record['iteration'] != iteration or record['db_path'] != str(args.db_path.resolve()):
        raise RuntimeError('Saved Amber run belongs to a different project/iteration')
    if record['settings'] != args.amber_remote or record['charge_method'] != args.amber_charge_method:
        raise RuntimeError('Amber settings changed during a saved run; recover with its original settings')
    if not record['prepared']:
        prepare_remote(args, iteration, directory, record)
    verify_sources(args, iteration, record)
    identities = database.amber_identity_rows(args.db_path, iteration, record['compound_ids'])
    saved_input = json.loads((directory / record['run_id'] / 'input.json').read_text())
    if saved_input['identities'] != [list(r) for r in identities]:
        raise RuntimeError('Compound identity changed during saved Amber work')
    remote = shlex.quote(record['remote_dir'])
    response = ssh(record['settings'], environment_command(record['settings'],
                   f'python {remote}/worker.py collect {remote}'), capture_output=True, text=True)
    snapshot = json.loads(response.stdout)
    save(directory / 'remote_snapshot.json', snapshot)
    if not snapshot['active'] and snapshot['status'] == 'failed':
        download_logs(directory, record)
        raise RuntimeError(snapshot.get('error') or 'Amber receptor preparation failed')
    results = snapshot['results']
    database.store_amber_results(args.db_path, iteration, results, record['compound_ids'])
    if snapshot['active']:
        return False
    completed = {r['id'] for r in results}
    if snapshot['status'] != 'done' or completed != set(record['compound_ids']):
        launch(record)
        return False
    # A terminal marker plus a released inherited lock establishes completion.
    record['finished'] = True
    download_logs(directory, record)
    save(directory / f"{record['run_id']}.completed.json", record)
    (directory / 'amber_job.json').unlink()
    return True


def download_logs(directory, record):
    remote = shlex.quote(record['remote_dir'])
    folder = directory / record['run_id']
    archive = folder / 'results.tar.pending'
    try:
        with archive.open('wb') as stream:
            ssh(record['settings'], environment_command(record['settings'],
                f'python {remote}/worker.py archive {remote}'), stdout=stream)
        allowed = {'calculation.log'} | {
            f'CMPID_{cid}/{name}' for cid in record['compound_ids']
            for name in ('calculation.log', 'FINAL_RESULTS_MMPBSA.dat')}
        with tarfile.open(archive) as stream:
            for member in stream:
                if member.name not in allowed or not member.isfile():
                    raise ValueError(f'Unexpected Amber artifact: {member.name}')
                destination = folder / member.name
                destination.parent.mkdir(exist_ok=True)
                with stream.extractfile(member) as source:
                    pending = destination.with_suffix(destination.suffix + '.pending')
                    pending.write_bytes(source.read())
                    pending.replace(destination)
    finally:
        archive.unlink(missing_ok=True)


def write_rankings(args, iteration, directory):
    rows = database.amber_rankings(args.db_path, iteration)
    selections = [('amber_scores.tsv', rows), ('amber_top.tsv', rows[:args.amber_keep_count]),
                  ('prime_candidates.tsv', rows[:args.amber_keep_count][:args.gbsa_input_count])]
    for name, selected in selections:
        pending = directory / (name + '.pending')
        with pending.open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['id', 'result_title', 'name', 'docking_score', 'amber_gbsa_score', 'gbsa_score'])
            writer.writerows((cid, f'CMPID_{cid}', name, dock, amber, prime)
                             for cid, name, dock, amber, prime in selected)
        os.replace(pending, directory / name)
    print(f'Iteration {iteration}: Amber finite scores={len(rows)}, retained={min(len(rows), args.amber_keep_count)}, '
          f'Prime candidates={min(len(rows), args.gbsa_input_count)}.')


def new_record(args, iteration, directory, ids):
    run_id = 'run-' + uuid.uuid4().hex
    record = {'run_id': run_id, 'compound_ids': ids, 'prepared': False,
              'iteration': iteration, 'db_path': str(args.db_path.resolve()),
              'settings': args.amber_remote, 'charge_method': args.amber_charge_method,
              'remote_dir': str(PurePosixPath(args.amber_remote['remote_work_dir']) / run_id)}
    save(directory / 'amber_job.json', record)
    return record
