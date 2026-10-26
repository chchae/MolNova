"""SSH dispatch tests do not submit scientific calculations."""
from types import SimpleNamespace
import shlex
from unittest.mock import Mock
import subprocess

import pytest
from molnova.stages import combichem_remote as remote


def config():
    return {'execution-host': 't41', 'remote-env-script': '/env with spaces/amber.sh',
            'config-path': '/shared/project with spaces/egfr.toml'}


def flags():
    return SimpleNamespace(export_only=True, debug=True, trace=True)


def test_dispatch_forwards_flags_and_quotes_shared_paths(monkeypatch):
    monkeypatch.setattr(remote.socket, 'gethostname', lambda: 'tensor.example')
    monkeypatch.delenv('MOLNOVA_COMBICHEM_REMOTE', raising=False)
    run = Mock()
    monkeypatch.setattr(remote.subprocess, 'run', run)
    assert remote.dispatch(config(), flags())
    args = run.call_args.args[0]
    assert args[:6] == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 't41']
    script = shlex.split(args[6])[2]
    assert "source '/env with spaces/amber.sh'" in script
    assert "'/shared/project with spaces/egfr.toml'" in script
    assert '--export-only --debug --trace' in script
    assert 'MOLNOVA_COMBICHEM_REMOTE=1' in script
    assert run.call_args.kwargs == {'check': True}


@pytest.mark.parametrize('host,marker', [('t41.example', None), ('alias-target', '1')])
def test_remote_host_or_dispatch_marker_prevents_recursion(monkeypatch, host, marker):
    monkeypatch.setattr(remote.socket, 'gethostname', lambda: host)
    monkeypatch.delenv('MOLNOVA_COMBICHEM_REMOTE', raising=False)
    if marker:
        monkeypatch.setenv('MOLNOVA_COMBICHEM_REMOTE', marker)
    run = Mock()
    monkeypatch.setattr(remote.subprocess, 'run', run)
    assert not remote.dispatch(config(), flags())
    run.assert_not_called()


def test_no_host_keeps_local_execution(monkeypatch):
    run = Mock()
    monkeypatch.setattr(remote.subprocess, 'run', run)
    assert not remote.dispatch({}, flags())
    run.assert_not_called()


def test_ssh_failure_propagates(monkeypatch):
    monkeypatch.setattr(remote.socket, 'gethostname', lambda: 'tensor')
    monkeypatch.delenv('MOLNOVA_COMBICHEM_REMOTE', raising=False)
    monkeypatch.setattr(remote.subprocess, 'run', Mock(side_effect=subprocess.CalledProcessError(255, 'ssh')))
    with pytest.raises(subprocess.CalledProcessError):
        remote.dispatch(config(), flags())


def test_distributed_install_is_reversible():
    from molnova.stages import combichem_3d as stage
    from molnova.stages import combichem_distributed as distributed
    original = distributed._ORIGINAL_WORKER or stage.worker_loop
    try:
        distributed.install(stage)
        assert stage.worker_loop is distributed.worker_loop
        distributed.uninstall(stage)
        assert stage.worker_loop is original
    finally:
        distributed.uninstall(stage)


def test_transport_round_trip_keeps_optimized_coordinates_and_atom_history():
    import io
    import json
    import numpy as np
    from rdkit import Chem
    from molnova.stages.combichem_distributed import RemoteSearch
    from molnova.chemistry.combichem3d import history as provenance
    mol = Chem.MolFromSmiles('CC')
    conf = Chem.Conformer(2)
    conf.SetAtomPosition(0, (0.123456789, 0., 0.))
    conf.SetAtomPosition(1, (1.654321987, 0., 0.))
    mol.AddConformer(conf)
    history = {'iteration': 1, 'atoms': [dict(index=i, uid=f'a{i}', branch='b',
                 fragment='f', added_iteration=1) for i in range(2)]}
    mol = provenance.restore(mol, history)
    optimized = [[.123456789, .234567891, .345678912], [1.712345678, .4, .5]]
    reply = dict(status='PASS', molblock=Chem.MolToMolBlock(mol), coordinates=optimized)
    remote = RemoteSearch('cpu2', {})
    remote.process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(json.dumps(reply)+'\n'))
    result = remote(mol, history, None, {}, 37)
    request = json.loads(remote.process.stdin.getvalue())
    np.testing.assert_array_equal(request['history']['pose_coordinates'], conf.GetPositions())
    np.testing.assert_array_equal(result['pose'].GetConformer().GetPositions(), optimized)
    assert result['pose'].GetAtomWithIdx(1).GetProp('_mn_uid') == 'a1'
    assert result['pose'].GetAtomWithIdx(1).GetIntProp('molnova_added_cycle') == 1


@pytest.mark.parametrize('response', ['', 'invalid JSON\n', '{"molblock":"invalid"}\n'])
def test_transport_failure_is_not_a_scientific_result(response):
    import io
    from rdkit import Chem
    from molnova.stages.combichem_distributed import RemoteSearch, RemoteCalculationError
    mol = Chem.MolFromSmiles('C'); mol.AddConformer(Chem.Conformer(1))
    remote = RemoteSearch('cpu2', {})
    remote.process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(response))
    with pytest.raises(RemoteCalculationError):
        remote(mol, {'iteration':0, 'atoms':[]}, None, {}, 37)


def test_distributed_dispatch_fingerprints_calculator_adapter(monkeypatch):
    from molnova.stages import combichem_3d as stage
    from molnova.stages import combichem_distributed as distributed
    cfg = config() | {'execution-host':'cpu1', 'worker-hosts':{'cpu1':192, 'cpu2':192}}
    monkeypatch.setattr(remote.socket, 'gethostname', lambda: 'cpu1')
    try:
        assert not remote.dispatch(cfg, flags())
        assert set(cfg['remote-implementation-sha256']) == {
            'combichem_remote.py', 'combichem_distributed.py', 'combichem_compute.py'}
        assert all(len(value)==64 for value in cfg['remote-implementation-sha256'].values())
    finally:
        distributed.uninstall(stage)


def test_calculator_connections_disable_inherited_ssh_multiplexing(monkeypatch, tmp_path):
    import io
    from molnova.stages import combichem_distributed as distributed
    process = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO('{"ready": true}\n'))
    popen = Mock(return_value=process)
    monkeypatch.setattr(distributed.subprocess, 'Popen', popen)
    monkeypatch.setattr(distributed.tempfile, 'gettempdir', lambda: str(tmp_path))
    cfg = {'remote-env-script':'/env/amber.sh', 'worker-ssh-known-hosts':'/shared/known_hosts'}
    distributed.RemoteSearch('cpu2', cfg).start()
    args = popen.call_args.args[0]
    for setting in ['ControlMaster=no', 'ControlPath=none', 'ControlPersist=no',
                    'BatchMode=yes', 'StrictHostKeyChecking=yes',
                    'UserKnownHostsFile=/shared/known_hosts', 'ServerAliveInterval=30']:
        assert args[args.index(setting)-1] == '-o'
    assert args[-2] == 'cpu2'
