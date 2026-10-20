"""SLURM admission and restart tests; no external submissions/calculations."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from molnova import _core, amber_gbsa as amber, amber_slurm as slurm, database
from molnova import amber_slurm_control as control
from molnova.config import amber_settings


PROJECT = '''
sidechain-library = "edges.smi"
dock-grid = "grid.zip"
reference-pose = "reference.maegz"
out-dir = "output"
target-count = 10000
sample-size = 100000
max-iteration = 100
host = "t41-cpu:128"
ligprep-cpus = 124
glide-cpus = 124
synthetic-feasibility-enabled = false
amber_gbsa = true
amber-cpus = 124
amber-ssh-host = "t41"
fep-input-count = 20
'''


@pytest.mark.parametrize('extra,debug', [('', False), ('debug = false\n', False),
                                          ('debug = true\n', True)])
def test_debug_population_limits_and_unchanged_scientific_resources(tmp_path, extra, debug):
    toml = tmp_path / 'test.toml'
    toml.write_text(PROJECT + extra)
    args = _core.load_project_toml(toml)
    assert [args[k] for k in ('target_count', 'sample_size', 'amber_keep_count',
                              'gbsa_input_count', 'gbsa_elite_count', 'fep_input_count')] == (
        [1000, 10000, 50, 5, 1, 2] if debug else [10000, 100000, 500, 50, 10, 20])
    assert args['debug'] is debug
    assert args['amber_remote']['host'] == 't41'
    assert args['use_slurm'] is False
    assert args['host'] == 't41-cpu:128'
    assert args['ligprep_cpus'] == args['glide_cpus'] == args['amber_cpus'] == 124
    assert args['max_iteration'] == 100
    assert args['ligprep_max_states'] == 2


def test_debug_explicit_counts_alias_and_floor_minimum(tmp_path):
    toml = tmp_path / 'test.toml'
    toml.write_text(PROJECT + 'debug = true\namber-keep-count = 123\n'
                    'gbsa-input-count = 19\ngbsa-elite-count = 1\nuse-slurm = true\n')
    args = _core.load_project_toml(toml)
    assert args['amber_keep_count'] == 12
    assert args['gbsa_input_count'] == args['gbsa_elite_count'] == 1
    assert args['amber_cpus'] == 384
    assert args['host'] == 't41-cpu:128'
    # File remains unmodified, so a separate stage scales exactly once.
    assert _core.load_project_toml(toml) == args


@pytest.mark.parametrize('key', ['use-slurm', 'debug'])
@pytest.mark.parametrize('value', ['1', '"true"', '0'])
def test_invalid_flags(tmp_path, key, value):
    toml = tmp_path / 'test.toml'
    toml.write_text(PROJECT + f'{key} = {value}\n')
    with pytest.raises(ValueError, match=key):
        _core.load_project_toml(toml)


@pytest.mark.parametrize('cpus', [True, 0, -1, 385, 384.0, '384'])
def test_slurm_budget_validation(tmp_path, cpus):
    with pytest.raises(ValueError, match='amber-slurm-cpus'):
        amber_settings({'out-dir': 'output', 'use-slurm': True,
                        'amber-slurm-cpus': cpus}, tmp_path)


def test_slurm_configuration_and_shared_environment(tmp_path):
    settings = amber_settings({'out-dir': 'output', 'use-slurm': True,
                               'amber-ssh-host': 't41', 'amber-cpus': 124}, tmp_path)
    assert settings['amber_cpus'] == 384
    remote = settings['amber_remote']
    assert remote['host'] == 'slurm' and remote['partition'] == 'cpu'
    assert remote['remote_work_dir'].startswith('/homes1/')
    assert remote['remote_work_dir'].endswith('/molnova-amber-slurm-runs')
    assert 'conda_init' not in remote
    command = amber.environment_command(remote, 'python worker.py')
    assert '/share/opt/amber26/envs/AmberTools26/amber.sh' in command
    assert 'conda activate' not in command
    assert 'OMP_NUM_THREADS=1' in command


@pytest.fixture
def project(tmp_path):
    db = tmp_path / 'test.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany("INSERT INTO compound(id,name,smiles,iteration,state,docking_score) "
                         "VALUES (?,?,'CCO',1,'amber_running',-1)",
                         [(i, f'GEN_{i}') for i in range(1, 5)])
    receptor = tmp_path / 'receptor.maegz'
    receptor.touch()
    poses = tmp_path / 'iter1/glide/best_poses.maegz'
    poses.parent.mkdir(parents=True)
    poses.touch()
    return SimpleNamespace(db_path=db, output=tmp_path, amber_receptor=receptor,
        amber_charge_method='bcc', amber_cpus=384, use_slurm=True,
        amber_remote={'host': 'slurm', 'partition': 'cpu', 'remote_env': '/shared/env',
                      'remote_work_dir': '/shared/runs'})


@pytest.mark.parametrize('count,budget,expected', [(500, 384, [192, 192]),
    (50, 384, [50]), (500, 193, [97, 96]), (500, 192, [192]), (1, 384, [1])])
def test_partitioning_caps_total_and_preserves_every_id(project, count, budget, expected):
    project.amber_cpus = budget
    record = amber.new_record(project, 1, project.output, list(range(1, count + 1)))
    assert record['mode'] == slurm.MODE
    assert [p['cpus'] for p in record['parts']] == expected
    assert sum(expected) <= 384
    assert sorted(cid for p in record['parts'] for cid in p['compound_ids']) == list(range(1, count + 1))
    assert len({p['remote_dir'] for p in record['parts']}) == len(expected)
    assert amber.load_job(project.output) == record


def child_record(project):
    record = slurm.new_record(project, 1, project.output, [1, 2, 3, 4])['parts'][0]
    record['prepared'] = True
    record['slurm_job_id'] = '123'
    folder = project.output / record['run_id']
    folder.mkdir()
    amber.save(folder / 'input.json', {'identities': database.amber_identity_rows(project.db_path, 1, [1,2,3,4])})
    amber.save(project.output / 'amber_job.json', record)
    return record


def test_queued_or_terminal_slurm_does_not_bypass_worker_completion(project, monkeypatch):
    record = child_record(project)
    native = {'active': True, 'state': 'PENDING'}
    snapshot = {'active': False, 'status': 'done',
                'results': [{'id': i, 'score': -i} for i in range(1, 5)]}
    monkeypatch.setattr(slurm, 'job_state', lambda *a: native)
    monkeypatch.setattr(amber, 'ssh', lambda *a, **kw: SimpleNamespace(stdout=json.dumps(snapshot)))
    monkeypatch.setattr(amber, 'launch', lambda *a: pytest.fail('Direct launch of Slurm job'))
    monkeypatch.setattr(amber, 'download_logs', lambda *a: None)
    assert not amber.poll(project, 1, project.output, record)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT COUNT(amber_gbsa_score) FROM compound').fetchone()[0] == 4
    assert (project.output / 'amber_job.json').exists()
    native.update(active=False, state='COMPLETED', successful=True)
    snapshot['active'] = True
    assert not amber.poll(project, 1, project.output, record)
    snapshot['active'] = False
    assert amber.poll(project, 1, project.output, record)
    assert not (project.output / 'amber_job.json').exists()


@pytest.mark.parametrize('successful,state', [(False, 'FAILED'), (True, 'COMPLETED')])
def test_incomplete_terminal_slurm_retains_identity_and_claims(project, monkeypatch, successful, state):
    record = child_record(project)
    monkeypatch.setattr(slurm, 'job_state', lambda *a: dict(active=False, successful=successful, state=state))
    monkeypatch.setattr(amber, 'ssh', lambda *a, **kw: SimpleNamespace(stdout=json.dumps(
        dict(active=False, status=None, results=[{'id':1, 'score':-1}]))))
    monkeypatch.setattr(amber, 'launch', lambda *a: pytest.fail('Resubmission'))
    monkeypatch.setattr(amber, 'download_logs', lambda *a: None)
    with pytest.raises(RuntimeError, match='retained'):
        amber.poll(project, 1, project.output, record)
    assert (project.output / 'amber_job.json').exists()
    with database.connect(project.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM compound WHERE state='amber_running'").fetchone()[0] == 3
        assert conn.execute('SELECT amber_gbsa_score FROM compound WHERE id=1').fetchone()[0] == -1


def test_lost_sbatch_response_recovers_exact_job_without_duplicate(tmp_path, monkeypatch):
    amber.save(tmp_path / 'input.json', {'cpus': 192, 'slurm': {'partition': 'cpu', 'job_name': 'unique'}})
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 90)
    monkeypatch.setattr(control.subprocess, 'check_output', submit)
    with pytest.raises(subprocess.TimeoutExpired):
        control.submit(tmp_path)
    assert json.loads((tmp_path / 'slurm-submission.json').read_text())['launching']
    assert calls[0][:2] == ['sbatch', '--parsable']
    assert calls[0][calls[0].index('--partition') + 1] == 'cpu'
    queries = []
    def query(command):
        queries.append(command)
        return '' if command[0] == 'squeue' else '123|unique|\n'
    monkeypatch.setattr(control, 'query', query)
    record = control.submit(tmp_path)
    assert record['job_id'] == '123' and not record['launching']
    assert len(calls) == 1
    assert {q[0] for q in queries} == {'squeue', 'sacct'}


def test_unknown_history_never_retries_sbatch(tmp_path, monkeypatch):
    amber.save(tmp_path / 'slurm-submission.json', dict(launching=True, started_at=0, job_name='unique'))
    monkeypatch.setattr(control, 'query', lambda *a: (_ for _ in ()).throw(RuntimeError('accounting unavailable')))
    monkeypatch.setattr(control.subprocess, 'check_output', lambda *a, **k: pytest.fail('Duplicate sbatch'))
    with pytest.raises(RuntimeError, match='unavailable'):
        control.submit(tmp_path)
    assert json.loads((tmp_path / 'slurm-submission.json').read_text())['launching']


@pytest.mark.parametrize('rows,active,success', [
    ('123|COMPLETED|\n123.batch|COMPLETED|\n123.0|COMPLETED|', False, True),
    ('123|COMPLETED|\n123.0|RUNNING|', True, False),
    ('123|COMPLETED|\n123.0|FAILED|', False, False),
])
def test_slurm_parent_and_steps_must_finish(monkeypatch, rows, active, success):
    monkeypatch.setattr(control, 'query', lambda c: '' if c[0] == 'squeue' else rows)
    result = control.status('123')
    assert result['active'] is active and result['successful'] is success


def test_missing_accounting_is_not_completion(monkeypatch):
    monkeypatch.setattr(control, 'query', lambda c: '')
    with pytest.raises(RuntimeError, match='no verified status'):
        control.status('123')


def test_preparation_uploads_slurm_script_and_pins_worker_count(project, monkeypatch):
    import tarfile
    record = child_record(project)
    record['prepared'] = False
    folder = project.output / record['run_id']
    def export(args, iteration, root, ids):
        for name in ('receptor.pdb', 'receptor.normalized.pdb', 'receptor_bonds.in'):
            (root / name).touch()
        for cid in ids:
            (root / f'CMPID_{cid}').mkdir()
            (root / f'CMPID_{cid}/input.mol2').touch()
        return [{'id': cid, 'charge': 0} for cid in ids]
    monkeypatch.setattr(amber, 'export_inputs', export)
    monkeypatch.setattr(amber, 'ssh', lambda *a, **kw: None)
    project.amber_cpus = record['cpus']
    amber.prepare_remote(project, 1, project.output, record)
    config = json.loads((folder / 'input.json').read_text())
    assert config['cpus'] == 4
    assert config['slurm']['partition'] == 'cpu'
    assert config['slurm']['job_name'].endswith(record['run_id'])
    script = (folder / 'amber.sbatch').read_text()
    assert 'srun --ntasks=1 --cpus-per-task=4' in script
    assert '/shared/env/amber.sh' in script
    with tarfile.open(folder / 'inputs.tar') as archive:
        assert {'amber.sbatch', 'amber_slurm_control.py', 'worker.py', 'input.json'} <= set(archive.getnames())
    assert amber.load_job(project.output)['prepared']


def test_parent_collects_other_shard_even_when_one_is_uncertain(project, monkeypatch):
    record = slurm.new_record(project, 1, project.output, list(range(1, 501)))
    calls = []
    def poll(args, iteration, directory, child):
        calls.append(child['run_id'])
        if child['run_id'] == record['parts'][0]['run_id']:
            raise RuntimeError('unknown Slurm status')
        amber.save(directory / (child['run_id'] + '.completed.json'), child)
        (directory / 'amber_job.json').unlink()
        return True
    monkeypatch.setattr(amber, 'poll', poll)
    with pytest.raises(RuntimeError, match='recovery pending'):
        slurm.poll(project, 1, project.output, record)
    assert calls == [p['run_id'] for p in record['parts']]
    assert (project.output / 'amber_job.json').exists()
    # Restart must not poll or submit the already retired second shard.
    monkeypatch.setattr(amber, 'poll', lambda *a: True)
    assert slurm.poll(project, 1, project.output, record)
    assert not (project.output / 'amber_job.json').exists()
