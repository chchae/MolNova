"""Isolated one-compound array tasks; copied beside the unchanged Amber worker."""
import json
from pathlib import Path
import sys
import tarfile

if __package__:
    from . import amber_runner as worker
else:
    import worker


def prepare_tasks(root, config):
    """Create immutable inputs locally; tasks share only read-only source files."""
    for index, entry in enumerate(config['entries']):
        task = root / 'tasks' / str(index)
        task.mkdir(parents=True, exist_ok=True)
        settings = dict(config, cpus=1, entries=[entry])
        settings.pop('slurm', None)
        worker.save(task / 'input.json', settings)
        for name in ('receptor.pdb', 'receptor_bonds.in'):
            link = task / name
            if not link.is_symlink():
                link.symlink_to('../../' + name)
        ligand = task / f'CMPID_{entry["id"]}'
        ligand.mkdir(exist_ok=True)
        link = ligand / 'input.mol2'
        if not link.is_symlink():
            link.symlink_to(f'../../../CMPID_{entry["id"]}/input.mol2')


def entries(root):
    return json.loads((root / 'input.json').read_text())['entries']


def execute(root, index):
    manifest = entries(root)
    if not 0 <= index < len(manifest):
        raise ValueError('Amber array task index outside saved manifest')
    task = root / 'tasks' / str(index)
    config = json.loads((task / 'input.json').read_text())
    if config['entries'] != [manifest[index]] or config['cpus'] != 1:
        raise ValueError('Amber array task input differs from saved manifest')
    worker.execute(task)


def snapshot(root):
    snapshots, results, errors = [], [], []
    for index, entry in enumerate(entries(root)):
        current = worker.snapshot(root / 'tasks' / str(index))
        if any(result['id'] != entry['id'] for result in current['results']):
            raise ValueError('Amber array task returned an unexpected compound identity')
        snapshots.append(current)
        results.extend(current['results'])
        if current['status'] == 'failed':
            errors.append(f'CMPID_{entry["id"]}: {current["error"]}')
    return {'active': any(value['active'] for value in snapshots),
            'status': 'failed' if errors else
                      'done' if all(value['status'] == 'done' for value in snapshots) else 'running',
            'error': '; '.join(errors) or None, 'results': results,
            'tasks': snapshots}


def archive_logs(root):
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        for index, entry in enumerate(entries(root)):
            task = root / 'tasks' / str(index)
            prefix = f'CMPID_{entry["id"]}'
            for name in ('calculation.log', 'minimize.in', 'minimize.out', 'FINAL_RESULTS_MMPBSA.dat'):
                path = task / prefix / name
                if path.is_file():
                    archive.add(path, arcname=f'{prefix}/{name}')
            if (task / 'calculation.log').is_file():
                archive.add(task / 'calculation.log', arcname=f'{prefix}/receptor_calculation.log')


if __name__ == '__main__':
    root = Path(sys.argv[2]).resolve()
    if sys.argv[1] == 'collect':
        print(json.dumps(snapshot(root)))
    elif sys.argv[1] == 'archive':
        archive_logs(root)
    else:
        execute(root, int(sys.argv[3]))
