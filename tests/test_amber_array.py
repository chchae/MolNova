"""Array task isolation and incremental recovery; no external calculations."""
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from molnova import amber_array as array, amber_runner as worker


@pytest.fixture
def run(tmp_path):
    config = {'entries': [{'id': 11, 'charge': 0}, {'id': 22, 'charge': -1}],
              'cpus': 384, 'charge_method': 'bcc', 'protocol': worker.PROTOCOL,
              'minimization': worker.MINIMIZATION, 'slurm': {'array_count': 2}}
    worker.save(tmp_path / 'input.json', config)
    (tmp_path / 'receptor.pdb').write_text('original receptor')
    (tmp_path / 'receptor_bonds.in').write_text('original bonds')
    for entry in config['entries']:
        ligand = tmp_path / f'CMPID_{entry["id"]}'
        ligand.mkdir()
        (ligand / 'input.mol2').write_text(f'pose {entry["id"]}')
    array.prepare_tasks(tmp_path, config)
    return tmp_path


def test_tasks_execute_only_their_compound_and_preserve_scientific_settings(run, monkeypatch):
    calls = []
    def execute(task):
        settings = json.loads((task / 'input.json').read_text())
        assert settings['cpus'] == 1
        assert settings['protocol'] == worker.PROTOCOL
        assert settings['minimization'] == worker.MINIMIZATION
        assert settings['charge_method'] == 'bcc'
        assert len(settings['entries']) == 1
        entry = settings['entries'][0]
        calls.append(entry['id'])
        assert (task / 'receptor.pdb').read_text() == 'original receptor'
        assert (task / 'receptor_bonds.in').read_text() == 'original bonds'
        assert (task / f'CMPID_{entry["id"]}/input.mol2').read_text() == f'pose {entry["id"]}'
        (task / 'receptor.normalized.pdb').write_text(f'normalized for {entry["id"]}')
    monkeypatch.setattr(worker, 'execute', execute)
    array.execute(run, 1)
    array.execute(run, 0)
    assert calls == [22, 11]
    assert (run / 'receptor.pdb').read_text() == 'original receptor'
    assert not (run / 'receptor.normalized.pdb').exists()
    assert (run / 'tasks/0/receptor.normalized.pdb').read_text() == 'normalized for 11'


def test_missing_or_changed_task_input_never_runs_another_compound(run, monkeypatch):
    monkeypatch.setattr(worker, 'execute', lambda *a: pytest.fail('Wrong task executed'))
    with pytest.raises(ValueError, match='index'):
        array.execute(run, -1)
    with pytest.raises(ValueError, match='index'):
        array.execute(run, 2)
    config = json.loads((run / 'tasks/0/input.json').read_text())
    config['entries'] = [{'id': 22, 'charge': -1}]
    worker.save(run / 'tasks/0/input.json', config)
    with pytest.raises(ValueError, match='differs'):
        array.execute(run, 0)


def test_array_collects_partial_results_but_requires_every_marker_and_lock_release(run):
    first, second = run / 'tasks/0', run / 'tasks/1'
    (first / 'results').mkdir()
    worker.save(first / 'results/CMPID_11.json', {'id': 11, 'score': -20})
    worker.save(first / 'status.json', {'status': 'done'})
    partial = array.snapshot(run)
    assert partial['results'] == [{'id': 11, 'score': -20}]
    assert partial['status'] == 'running'
    (second / 'results').mkdir()
    worker.save(second / 'results/CMPID_22.json', {'id': 22, 'error': 'parameterization failed'})
    assert array.snapshot(run)['status'] == 'running'
    worker.save(second / 'status.json', {'status': 'done'})
    with (second / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert array.snapshot(run)['active']
    complete = array.snapshot(run)
    assert not complete['active'] and complete['status'] == 'done'
    assert len(complete['results']) == 2


def test_preparation_failure_preserves_other_task_results(run):
    task = run / 'tasks/0'
    worker.save(task / 'status.json', {'status': 'failed', 'error': 'receptor error'})
    other = run / 'tasks/1'
    (other / 'results').mkdir()
    worker.save(other / 'results/CMPID_22.json', {'id': 22, 'score': -15})
    current = array.snapshot(run)
    assert current['status'] == 'failed'
    assert 'CMPID_11: receptor error' == current['error']
    assert current['results'] == [{'id': 22, 'score': -15}]


def test_wrong_task_result_is_rejected(run):
    results = run / 'tasks/0/results'
    results.mkdir()
    worker.save(results / 'CMPID_22.json', {'id': 22, 'score': -15})
    with pytest.raises(ValueError, match='identity'):
        array.snapshot(run)


def test_array_archive_maps_logs_to_original_compound_ids(run, monkeypatch):
    (run / 'tasks/0/CMPID_11/calculation.log').write_text('ligand log')
    (run / 'tasks/0/calculation.log').write_text('receptor log')
    (run / 'tasks/1/CMPID_22/minimize.out').write_text('minimization log')
    stream = io.BytesIO()
    monkeypatch.setattr(sys, 'stdout', SimpleNamespace(buffer=stream))
    array.archive_logs(run)
    stream.seek(0)
    with tarfile.open(fileobj=stream) as archive:
        assert set(archive.getnames()) == {'CMPID_11/calculation.log',
                 'CMPID_11/receptor_calculation.log', 'CMPID_22/minimize.out'}
        assert archive.extractfile('CMPID_11/calculation.log').read() == b'ligand log'


def test_uploaded_standalone_helpers_collect_without_amber_activation(run):
    from molnova import receptor_pdb

    for source, name in [(Path(array.__file__), 'amber_array.py'),
                         (Path(worker.__file__), 'worker.py'),
                         (Path(receptor_pdb.__file__), 'receptor_pdb.py')]:
        (run / name).write_bytes(source.read_bytes())
    response = subprocess.run([sys.executable, str(run / 'amber_array.py'), 'collect', str(run)],
                              check=True, capture_output=True, text=True)
    assert json.loads(response.stdout) == dict(active=False, status='running', error=None, results=[],
        tasks=[dict(active=False, status=None, error=None, results=[]) for _ in range(2)])
    # The current collector also works through stdin beside an older saved worker.
    response = subprocess.run([sys.executable, '-', 'collect', str(run)], cwd=run,
                              input=Path(array.__file__).read_text(),
                              check=True, capture_output=True, text=True)
    assert len(json.loads(response.stdout)['tasks']) == 2


@pytest.mark.parametrize('policy', ['legacy', 'legacy_retry', 'checkpoint', 'checkpoint_charge'])
def test_actual_worker_accepts_frozen_protocol_and_reaches_calculation(run, monkeypatch, policy):
    # Exercise execute itself: mocking execute hid the input/worker mismatch.
    config = json.loads((run / 'input.json').read_text())
    if policy != 'legacy':
        config['minimization_timeout_s'] = 600
    if policy in ('checkpoint', 'checkpoint_charge'):
        config.update(protocol=worker.CHECKPOINT_PROTOCOL, minimization=worker.CHECKPOINT_MINIMIZATION)
    if policy == 'checkpoint_charge':
        config['charge_timeout_s'] = 600
    worker.save(run / 'input.json', config)
    array.prepare_tasks(run, config)
    calls = []
    monkeypatch.setattr(worker, 'normalize_receptor_pdb', lambda *a: None)
    monkeypatch.setattr(worker, 'run_leap', lambda *a: None)
    monkeypatch.setitem(sys.modules, 'parmed', SimpleNamespace(load_file=lambda *a, **k: object()))
    def calculate(root, entry, method, lock_fd, **options):
        calls.append((entry['id'], options))
        worker.save(root / 'results' / f"CMPID_{entry['id']}.json", {'id': entry['id'], 'score': -20})
    monkeypatch.setattr(worker, 'calculate', calculate)
    array.execute(run, 0)
    expected = {} if policy == 'legacy' else {'minimization_timeout_s': 600}
    if policy == 'checkpoint_charge':
        expected['charge_timeout_s'] = 600
    assert calls == [(11, expected)]
    assert worker.snapshot(run / 'tasks/0')['status'] == 'done'


@pytest.mark.parametrize('change', [
    {'protocol': 'unknown'},
    {'minimization': worker.MINIMIZATION},
    {'minimization': dict(worker.CHECKPOINT_MINIMIZATION, restraint_wt=0)},
    {'minimization_timeout_s': None}, {'minimization_timeout_s': True},
    {'minimization_timeout_s': -1}, {'minimization_timeout_s': float('nan')},
    {'minimization_timeout_s': float('inf')}, {'minimization_timeout_s': '600'},
])
def test_invalid_checkpoint_policy_stops_before_receptor_or_ligand_calculation(run, monkeypatch, change):
    task = run / 'tasks/0'
    config = json.loads((task / 'input.json').read_text())
    config.update(protocol=worker.CHECKPOINT_PROTOCOL, minimization=worker.CHECKPOINT_MINIMIZATION,
                  minimization_timeout_s=600)
    config.update(change)
    worker.save(task / 'input.json', config)
    monkeypatch.setattr(worker, 'normalize_receptor_pdb', lambda *a: pytest.fail('Invalid protocol reached receptor preparation'))
    monkeypatch.setattr(worker, 'calculate', lambda *a, **k: pytest.fail('Invalid protocol reached scientific calculation'))
    worker.execute(task)
    snapshot = worker.snapshot(task)
    assert snapshot['status'] == 'failed' and not snapshot['active']
    assert snapshot['results'] == []


@pytest.mark.parametrize('missing', ['minimization', 'minimization_timeout_s'])
def test_checkpoint_protocol_requires_explicit_settings(missing):
    config = dict(protocol=worker.CHECKPOINT_PROTOCOL, minimization=worker.CHECKPOINT_MINIMIZATION,
                  minimization_timeout_s=600)
    del config[missing]
    with pytest.raises(RuntimeError, match='requires pinned'):
        worker.validate_protocol(config)
