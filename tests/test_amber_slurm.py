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


@pytest.mark.parametrize('count,budget', [(500, 384), (50, 384), (500, 193), (500, 192), (1, 384)])
def test_array_record_preserves_task_order_and_cpu_limit(project, count, budget):
    project.amber_cpus = budget
    record = amber.new_record(project, 1, project.output, list(range(1, count + 1)))
    assert record['mode'] == slurm.ARRAY
    assert record['cpus'] == budget
    assert record['compound_ids'] == list(range(1, count + 1))
    assert not record['prepared']
    assert 'parts' not in record
    assert amber.load_job(project.output) == record


def legacy_record(project):
    """Saved pre-array format must remain recoverable without conversion."""
    ids = list(range(1, 501))
    parts = [dict(run_id=f'legacy-{index}', compound_ids=ids[index::2], cpus=192,
                  prepared=False, mode=slurm.SINGLE, iteration=1,
                  db_path=str(project.db_path.resolve()), settings=dict(project.amber_remote),
                  charge_method='bcc', remote_dir=f'/shared/runs/legacy-{index}') for index in range(2)]
    record = dict(mode=slurm.MODE, run_id='legacy-parent', iteration=1,
                  db_path=str(project.db_path.resolve()), settings=dict(project.amber_remote),
                  charge_method='bcc', compound_ids=ids, cpus=384, parts=parts)
    amber.save(project.output / 'amber_job.json', record)
    return record


def child_record(project, mode=slurm.SINGLE):
    if mode == slurm.SINGLE:
        record = legacy_record(project)['parts'][0]
        record.update(compound_ids=[1, 2, 3, 4], cpus=4)
    else:
        record = slurm.new_record(project, 1, project.output, [1, 2, 3, 4])
    record['prepared'] = True
    record['slurm_job_id'] = '123'
    folder = project.output / record['run_id']
    folder.mkdir()
    amber.save(folder / 'input.json', {'identities': database.amber_identity_rows(project.db_path, 1, [1,2,3,4])})
    amber.save(project.output / 'amber_job.json', record)
    return record


@pytest.mark.parametrize('mode', [slurm.SINGLE, slurm.ARRAY])
def test_queued_or_terminal_slurm_does_not_bypass_worker_completion(project, monkeypatch, mode):
    record = child_record(project, mode)
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
@pytest.mark.parametrize('mode', [slurm.SINGLE, slurm.ARRAY])
def test_incomplete_terminal_slurm_retains_identity_and_claims(project, monkeypatch, successful, state, mode):
    record = child_record(project, mode)
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


@pytest.mark.parametrize('mode', [slurm.SINGLE, slurm.ARRAY])
@pytest.mark.parametrize('saved_array', [False, True])
def test_preparation_uploads_slurm_script_and_pins_worker_count(project, monkeypatch, mode, saved_array):
    import tarfile
    record = child_record(project, mode)
    if mode == slurm.ARRAY:
        assert record['slurm_time_limit'] == '00:30:00'
        if saved_array:
            del record['slurm_time_limit']
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
    assert config['cpus'] == record['cpus']
    assert config['slurm']['partition'] == 'cpu'
    assert config['slurm']['job_name'].endswith(record['run_id'])
    if mode == slurm.ARRAY and not saved_array:
        assert config['slurm']['time_limit'] == '00:30:00'
    else:
        assert 'time_limit' not in config['slurm']
    script = (folder / 'amber.sbatch').read_text()
    assert f'srun --ntasks=1 --cpus-per-task={1 if mode == slurm.ARRAY else 4}' in script
    assert '/shared/env/amber.sh' in script
    with tarfile.open(folder / 'inputs.tar') as archive:
        assert {'amber.sbatch', 'amber_slurm_control.py', 'worker.py', 'input.json'} <= set(archive.getnames())
        if mode == slurm.ARRAY:
            assert config['slurm']['array_count'] == 4
            assert '"$SLURM_ARRAY_TASK_ID"' in script
            assert 'amber_array.py' in archive.getnames()
            for index, cid in enumerate(record['compound_ids']):
                task_config = json.load(archive.extractfile(f'tasks/{index}/input.json'))
                assert task_config['cpus'] == 1
                assert task_config['entries'] == [{'id': cid, 'charge': 0}]
                member = archive.getmember(f'tasks/{index}/CMPID_{cid}/input.mol2')
                assert member.issym() and member.linkname == f'../../../CMPID_{cid}/input.mol2'
    assert amber.load_job(project.output)['prepared']


def test_parent_collects_other_shard_even_when_one_is_uncertain(project, monkeypatch):
    record = legacy_record(project)
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


@pytest.mark.parametrize('count,cpus,throttle', [(50, 384, 50), (500, 384, 384), (4, 2, 2), (1, 1, 1)])
@pytest.mark.parametrize('time_limit', [None, '00:30:00'])
def test_array_submission_requests_one_core_and_bounded_memory(tmp_path, monkeypatch, count, cpus, throttle, time_limit):
    options = {'partition': 'cpu', 'job_name': 'unique', 'array_count': count}
    if time_limit is not None:
        options['time_limit'] = time_limit
    amber.save(tmp_path / 'input.json', {'cpus': cpus, 'entries': [{'id': i} for i in range(count)],
               'slurm': options})
    commands = []
    def submit(command, **kwargs):
        commands.append(command)
        return '123;cluster\n'
    monkeypatch.setattr(control.subprocess, 'check_output', submit)
    assert control.submit(tmp_path)['job_id'] == '123'
    command = commands[0]
    assert command[command.index('--cpus-per-task') + 1] == '1'
    assert command[command.index('--array') + 1] == f'0-{count - 1}%{throttle}'
    assert '--mem=2G' in command and '--mem=0' not in command
    assert [value for value in command if value.startswith('--time=')] == (
        [] if time_limit is None else ['--time=00:30:00'])
    assert str(tmp_path / 'slurm-%A_%a.out') in command
    assert str(tmp_path / 'slurm-%A_%a.err') in command
    assert control.submit(tmp_path)['job_id'] == '123'
    assert len(commands) == 1


def test_lost_array_response_recovers_base_id_from_tasks(tmp_path, monkeypatch):
    amber.save(tmp_path / 'input.json', {'cpus': 384, 'entries': [{'id': 1}, {'id': 2}],
               'slurm': {'partition': 'cpu', 'job_name': 'unique', 'array_count': 2}})
    def lost(*args, **kwargs):
        raise subprocess.TimeoutExpired('sbatch', 90)
    monkeypatch.setattr(control.subprocess, 'check_output', lost)
    with pytest.raises(subprocess.TimeoutExpired):
        control.submit(tmp_path)
    queries = []
    def query(command):
        queries.append(command)
        return '123\n123\n' if command[0] == 'squeue' else '123_0|unique|\n123_1|unique|\n'
    monkeypatch.setattr(control, 'query', query)
    assert control.submit(tmp_path)['job_id'] == '123'
    assert '--format=%F' in queries[0]
    assert '--array' in queries[1]


@pytest.mark.parametrize('history', ['123_0|unique|\n456_0|unique|', '123_[0-1]|unique|'])
def test_ambiguous_array_history_blocks_resubmission(tmp_path, monkeypatch, history):
    amber.save(tmp_path / 'slurm-submission.json', dict(launching=True, started_at=0,
                                                       job_name='unique', array_count=2))
    monkeypatch.setattr(control, 'query', lambda command: '' if command[0] == 'squeue' else history)
    monkeypatch.setattr(control.subprocess, 'check_output', lambda *a, **kw: pytest.fail('Duplicate array'))
    with pytest.raises(RuntimeError, match='resubmission blocked'):
        control.submit(tmp_path)


@pytest.mark.parametrize('rows,active,successful', [
    ('123_0|COMPLETED|\n123_0.batch|COMPLETED|\n123_1|COMPLETED|\n123_1.0|COMPLETED|', False, True),
    ('123_0|COMPLETED|\n123_1|COMPLETED|\n123_1.0|RUNNING|', True, False),
    ('123_0|COMPLETED|\n123_1|FAILED|', False, False),
    ('123_0|COMPLETED|\n123_1|CANCELLED by 1023|', False, False),
])
def test_array_status_requires_all_tasks_and_steps(monkeypatch, rows, active, successful):
    monkeypatch.setattr(control, 'query', lambda command: '' if command[0] == 'squeue' else rows)
    result = control.status('123', 2)
    assert result['active'] is active and result['successful'] is successful


@pytest.mark.parametrize('rows', ['', '123|COMPLETED|', '123_0|COMPLETED|',
                                  '123_0|COMPLETED|\n123_1|COMPLETED|\n123_2|COMPLETED|',
                                  '123_0|COMPLETED|\n123_1|COMPLETED|\n123_2.0|COMPLETED|'])
def test_missing_or_unexpected_array_task_history_never_completes(monkeypatch, rows):
    monkeypatch.setattr(control, 'query', lambda command: '' if command[0] == 'squeue' else rows)
    with pytest.raises(RuntimeError, match='task history'):
        control.status('123', 2)


def test_array_status_keeps_pending_tasks_without_accounting_lookup(monkeypatch):
    def query(command):
        assert command[0] == 'squeue'
        return 'RUNNING\nPENDING\n'
    monkeypatch.setattr(control, 'query', query)
    assert control.status('123', 2)['active']


def test_array_poll_persists_successes_before_reporting_other_task_preparation_failure(project, monkeypatch):
    record = child_record(project, slurm.ARRAY)
    monkeypatch.setattr(slurm, 'job_state', lambda *a: dict(active=False, successful=True, state='COMPLETED'))
    monkeypatch.setattr(amber, 'ssh', lambda *a, **kw: SimpleNamespace(stdout=json.dumps(
        dict(active=False, status='failed', error='receptor error', results=[{'id': 1, 'score': -20}]))))
    monkeypatch.setattr(amber, 'download_logs', lambda *a: None)
    with pytest.raises(RuntimeError, match='receptor error'):
        amber.poll(project, 1, project.output, record)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT amber_gbsa_score FROM compound WHERE id=1').fetchone() == (-20,)
    assert (project.output / 'amber_job.json').exists()


def test_array_job_state_checkpoints_id_before_status_lookup(project, monkeypatch):
    record = child_record(project, slurm.ARRAY)
    del record['slurm_job_id']
    calls = []
    def ssh(settings, command, **kwargs):
        calls.append(command)
        if ' submit ' in command:
            return SimpleNamespace(stdout='{"job_id":"321"}')
        assert amber.load_job(project.output)['slurm_job_id'] == '321'
        assert command.endswith(' status 321 4')
        raise RuntimeError('status unavailable')
    monkeypatch.setattr(amber, 'ssh', ssh)
    with pytest.raises(RuntimeError, match='unavailable'):
        slurm.job_state(project.output, record)
    with pytest.raises(RuntimeError, match='unavailable'):
        slurm.job_state(project.output, amber.load_job(project.output))
    assert sum(' submit ' in command for command in calls) == 1
