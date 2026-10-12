"""Standalone AmberTools compute helper, copied to the remote runtime.

No SQLite access or stage coordination: the local Amber stage owns all claims.
Remote file locks and atomic results make reconnect/relaunch idempotent.
"""
import concurrent.futures
import fcntl
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile

if __package__:
    from .receptor_pdb import CAP_TEMPLATE_NAMES, NORMALIZATION_VERSION, normalize_receptor_pdb
else:
    from receptor_pdb import CAP_TEMPLATE_NAMES, NORMALIZATION_VERSION, normalize_receptor_pdb


def save(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def run(command, folder, lock_fd):
    with (folder / 'calculation.log').open('ab') as log:
        log.write(('\n$ ' + ' '.join(command) + '\n').encode())
        log.flush()
        subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                       check=True, pass_fds=(lock_fd,))


def run_leap(script, folder, lock_fd, products):
    """A zero LEaP exit is insufficient: reject reported errors and missing outputs."""
    for name in products:
        (folder / name).unlink(missing_ok=True)
    log_path = folder / 'calculation.log'
    offset = log_path.stat().st_size if log_path.exists() else 0
    run(['tleap', '-f', script], folder, lock_fd)
    with log_path.open('rb') as log:
        log.seek(offset)
        output = log.read().decode('utf-8', errors='replace')
    if (re.search(r'FATAL|\bError\b|Failed to generate parameters', output, re.IGNORECASE)
            or re.search(r'Errors\s*=\s*[1-9]\d*', output)):
        raise RuntimeError('LEaP reported errors; see calculation.log')
    for name in products:
        path = folder / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f'LEaP did not create a complete {name}; see calculation.log')


def mol2_sections(path):
    lines = path.read_text().splitlines()
    section = None
    atoms, bonds = [], set()
    for index, line in enumerate(lines):
        if line.startswith('@<TRIPOS>'):
            section = line
        elif line.strip() and section == '@<TRIPOS>ATOM':
            atoms.append((index, line.split()))
        elif line.strip() and section == '@<TRIPOS>BOND':
            fields = line.split()
            bonds.add(tuple(sorted((int(fields[1]), int(fields[2])))))
    return lines, atoms, bonds


def restore_pose(original, typed):
    """Parameterization may relax SQM coordinates; always score the Glide pose."""
    _, source, source_bonds = mol2_sections(original)
    lines, target, target_bonds = mol2_sections(typed)
    if len(source) != len(target) or source_bonds != target_bonds:
        raise ValueError('Antechamber changed ligand atom order/connectivity')
    for (_, before), (index, after) in zip(source, target):
        if before[:2] != after[:2]:
            raise ValueError('Antechamber changed ligand atom identity')
        after[2:5] = before[2:5]
        lines[index] = ' '.join(after)
    typed.write_text('\n'.join(lines) + '\n')


def parse_score(path):
    matches = re.findall(r'^\s*DELTA TOTAL\s+(\S+)', path.read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError('Expected exactly one DELTA TOTAL in Amber GB results')
    score = float(matches[0])
    if not math.isfinite(score):
        raise ValueError('Non-finite Amber GBSA score')
    return score


def calculate(root, entry, charge_method, lock_fd):
    cid = entry['id']
    output = root / 'results' / f'CMPID_{cid}.json'
    if output.exists():
        return
    folder = root / f'CMPID_{cid}'
    try:
        for name in ('ligand.mol2', 'ligand.frcmod', 'complex.prmtop', 'complex.inpcrd',
                     'ligand.prmtop', 'ligand.inpcrd', 'receptor.prmtop', 'receptor.inpcrd',
                     'FINAL_RESULTS_MMPBSA.dat'):
            (folder / name).unlink(missing_ok=True)
        # No coordinate/bond guessing from SMILES: use the docked, protonated state.
        run(['antechamber', '-i', 'input.mol2', '-fi', 'mol2', '-o', 'ligand.mol2',
             '-fo', 'mol2', '-at', 'gaff2', '-c', charge_method, '-nc', str(entry['charge']),
             '-rn', 'LIG', '-seq', 'n', '-an', 'n', '-du', 'n', '-pf', 'y'], folder, lock_fd)
        restore_pose(folder / 'input.mol2', folder / 'ligand.mol2')
        run(['parmchk2', '-i', 'ligand.mol2', '-f', 'mol2', '-s', 'gaff2',
             '-o', 'ligand.frcmod'], folder, lock_fd)
        (folder / 'leap.in').write_text(
            'source leaprc.protein.ff14SB\nsource leaprc.gaff2\n'
            'source leaprc.water.tip3p\nset default PBRadii mbondi2\n'
            'loadamberparams ligand.frcmod\n' + CAP_TEMPLATE_NAMES +
            'rec = loadpdb ../receptor.normalized.pdb\n'
            + (root / 'receptor_bonds.in').read_text() +
            'lig = loadmol2 ligand.mol2\ncom = combine {rec lig}\n'
            'saveamberparm rec receptor.prmtop receptor.inpcrd\n'
            'saveamberparm lig ligand.prmtop ligand.inpcrd\n'
            'saveamberparm com complex.prmtop complex.inpcrd\nquit\n')
        run_leap('leap.in', folder, lock_fd,
                 ('receptor.prmtop', 'receptor.inpcrd', 'ligand.prmtop', 'ligand.inpcrd',
                  'complex.prmtop', 'complex.inpcrd'))
        # LEaP can exit zero with fatal topology errors. Validate its actual products.
        import parmed
        complex_structure = parmed.load_file(str(folder / 'complex.prmtop'),
                                            xyz=str(folder / 'complex.inpcrd'))
        ligand = parmed.load_file(str(folder / 'ligand.prmtop'), xyz=str(folder / 'ligand.inpcrd'))
        _, atoms, _ = mol2_sections(folder / 'input.mol2')
        if len(ligand.atoms) != len(atoms):
            raise ValueError('LEaP changed the ligand atom count')
        for amber_atom, (_, source) in zip(complex_structure.atoms[-len(atoms):], atoms):
            xyz = list(map(float, source[2:5]))
            if amber_atom.name != source[1] or any(abs(a-b) > 0.002 for a, b in
                    zip((amber_atom.xx, amber_atom.xy, amber_atom.xz), xyz)):
                raise ValueError('LEaP changed ligand identity or docking coordinates')
        (folder / 'mmpbsa.in').write_text(
            '&general\n startframe=1, endframe=1, interval=1, entropy=0, keep_files=1,\n/\n'
            '&gb\n igb=5, saltcon=0.15,\n/\n')
        run(['MMPBSA.py', '-O', '-i', 'mmpbsa.in', '-cp', 'complex.prmtop',
             '-rp', 'receptor.prmtop', '-lp', 'ligand.prmtop', '-y', 'complex.inpcrd',
             '-o', 'FINAL_RESULTS_MMPBSA.dat'], folder, lock_fd)
        result = {'id': cid, 'score': parse_score(folder / 'FINAL_RESULTS_MMPBSA.dat')}
    except Exception as exc:
        result = {'id': cid, 'error': f'{type(exc).__name__}: {exc}; see CMPID_{cid}/calculation.log'}
    save(output, result)


def execute(root):
    with (root / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        # Native subprocesses inherit the lock: a dead coordinator cannot lead
        # to duplicate calculation while any of its native programs still lives.
        config = json.loads((root / 'input.json').read_text())
        (root / 'results').mkdir(exist_ok=True)
        save(root / 'status.json', {'status': 'running'})
        # Common receptor errors must stop the batch, not fail every ligand.
        try:
            version = config.get('receptor_normalization_version')
            if version is not None and version != NORMALIZATION_VERSION:
                raise RuntimeError(f'Unsupported receptor normalization version: {version}')
            normalize_receptor_pdb(root / 'receptor.pdb', root / 'receptor.normalized.pdb',
                                  root / 'calculation.log')
            (root / 'receptor_leap.in').write_text(
                'source leaprc.protein.ff14SB\nsource leaprc.water.tip3p\n'
                'set default PBRadii mbondi2\n' + CAP_TEMPLATE_NAMES +
                'rec = loadpdb receptor.normalized.pdb\n'
                + (root / 'receptor_bonds.in').read_text() +
                'saveamberparm rec receptor.prmtop receptor.inpcrd\nquit\n')
            run_leap('receptor_leap.in', root, lock.fileno(), ('receptor.prmtop', 'receptor.inpcrd'))
            import parmed
            parmed.load_file(str(root / 'receptor.prmtop'), xyz=str(root / 'receptor.inpcrd'))
        except Exception as exc:
            with (root / 'calculation.log').open('a') as log:
                log.write(f'\nReceptor preparation failed: {type(exc).__name__}: {exc}\n')
            save(root / 'status.json', {'status': 'failed', 'error': f'Receptor preparation: {exc}; see calculation.log'})
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=config['cpus']) as pool:
            futures = [pool.submit(calculate, root, entry, config['charge_method'], lock.fileno())
                       for entry in config['entries']]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        save(root / 'status.json', {'status': 'done'})


def collect(root):
    with (root / 'run.lock').open('a') as lock:
        active = True
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = False
        except BlockingIOError:
            pass
        status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        results = [json.loads(p.read_text()) for p in sorted((root / 'results').glob('CMPID_*.json'))]
        print(json.dumps({'active': active, 'status': status.get('status'),
                          'error': status.get('error'), 'results': results}))


def archive_logs(root):
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        for path in sorted(root.glob('CMPID_*/*')):
            if path.name in ('calculation.log', 'FINAL_RESULTS_MMPBSA.dat'):
                archive.add(path, arcname=str(path.relative_to(root)))
        if (root / 'calculation.log').exists():
            archive.add(root / 'calculation.log', arcname='calculation.log')


if __name__ == '__main__':
    root = Path(sys.argv[2]).resolve()
    if sys.argv[1] == 'collect':
        collect(root)
    elif sys.argv[1] == 'archive':
        archive_logs(root)
    else:
        execute(root)
