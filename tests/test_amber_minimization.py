import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from molnova import amber_runner


FINAL = 'FINAL RESULTS\n\n NSTEP ENERGY RMS GMAX NAME NUMBER\n 500 -123.4 0.01 0.05 C1 1\n'


@pytest.mark.parametrize('failure', [None, 'missing_final', 'nan_energy', 'missing_restart',
                                   'empty_restart', 'nan_coordinates', 'bad_restart', 'native_error'])
def test_minimization_validates_final_frame_and_preserves_reference(tmp_path, monkeypatch, failure):
    original = tmp_path / 'complex.inpcrd'
    original.write_bytes(b'original Glide complex coordinates')
    restart = tmp_path / 'complex.minimized.rst7'
    restart.write_text('stale restart')
    def run(command, folder, lock_fd):
        assert command[0] == 'sander'
        assert command[command.index('-ref')+1] == 'complex.inpcrd'
        assert command[command.index('-c')+1] == 'complex.inpcrd'
        assert not restart.exists()
        mdin = (folder/'minimize.in').read_text()
        assert "ntr=1, restraint_wt=2.0, restraintmask='!@H='" in mdin
        if failure == 'native_error':
            raise subprocess.CalledProcessError(1, command)
        (folder/'minimize.out').write_text('partial log' if failure == 'missing_final' else
            FINAL.replace('-123.4','NaN') if failure == 'nan_energy' else FINAL)
        if failure != 'missing_restart':
            restart.write_text('' if failure == 'empty_restart' else 'new restart')
    def load_file(path, xyz):
        assert xyz == str(restart)
        if failure == 'bad_restart':
            raise ValueError('Invalid restart atom count')
        return SimpleNamespace(atoms=[SimpleNamespace(xx=float('nan') if failure == 'nan_coordinates' else 1,
                                                     xy=2, xz=3)])
    monkeypatch.setattr(amber_runner,'run',run)
    monkeypatch.setitem(sys.modules,'parmed',SimpleNamespace(load_file=load_file))
    if failure is None:
        assert amber_runner.minimize_complex(tmp_path,1) == restart
        assert 'minimization complete' in (tmp_path/'calculation.log').read_text()
    else:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            amber_runner.minimize_complex(tmp_path,1)
    assert original.read_bytes() == b'original Glide complex coordinates'


@pytest.mark.parametrize('charge_deadline', [None, 600])
@pytest.mark.parametrize('fail_minimization,deadline', [(False, None), (True, None),
    (False, 600), (True, 600), ('timeout', 600), ('no_checkpoint', 600)])
def test_calculate_scores_minimized_complex_and_never_scores_failed_minimization(tmp_path, monkeypatch, fail_minimization, deadline, charge_deadline):
    folder = tmp_path/'CMPID_1'
    folder.mkdir()
    (tmp_path/'results').mkdir()
    (tmp_path/'receptor_bonds.in').write_text('')
    source = '@<TRIPOS>ATOM\n1 C1 1.0 2.0 3.0 C.3 1 LIG 0\n2 H2 4.0 5.0 6.0 H 1 LIG 0\n@<TRIPOS>BOND\n1 1 2 1\n'
    (folder/'input.mol2').write_text(source)
    calls = []
    def run(command, folder, lock_fd, timeout=None, kill_process_group=False):
        calls.append(command)
        if command[0] == 'antechamber':
            assert timeout == charge_deadline
            assert kill_process_group is (charge_deadline is not None)
            (folder/'ligand.mol2').write_text(source)
        elif command[0] == 'tleap':
            assert 'rec = loadpdb ../receptor.normalized.pdb' in (folder/'leap.in').read_text()
            (folder/'calculation.log').write_text('Exiting LEaP: Errors = 0\n')
            for name in ('complex','ligand','receptor'):
                (folder/f'{name}.prmtop').write_text('topology')
                (folder/f'{name}.inpcrd').write_text('original coordinates')
        elif command[0] == 'sander':
            assert timeout == deadline
            if deadline is not None:
                assert 'ntwr=-10, ntxo=1' in (folder/'minimize.in').read_text()
            if fail_minimization in ('timeout', 'no_checkpoint'):
                if fail_minimization == 'timeout':
                    (folder/'complex.minimized.rst7_120').write_text('relaxed checkpoint')
                raise subprocess.TimeoutExpired(command, timeout)
            (folder/'minimize.out').write_text('partial output' if fail_minimization else FINAL)
            (folder/'complex.minimized.rst7').write_text('relaxed coordinates')
        elif command[0] == 'MMPBSA.py':
            assert command[command.index('-y')+1] == 'complex.minimized.rst7'
            if fail_minimization == 'timeout':
                assert (folder/'complex.minimized.rst7').read_text() == 'relaxed checkpoint'
            assert 'restraint' not in (folder/'mmpbsa.in').read_text()
            (folder/'FINAL_RESULTS_MMPBSA.dat').write_text('DELTA TOTAL -10.25\n')
    c = SimpleNamespace(name='C1',xx=1,xy=2,xz=3)
    h = SimpleNamespace(name='H2',xx=4,xy=5,xz=6)
    def load_file(path, xyz=None):
        if str(path).endswith('ligand.prmtop'):
            return SimpleNamespace(atoms=[c,h])
        if str(xyz).endswith('complex.minimized.rst7'):
            # Coordinate relaxation is allowed; it must not be reset to the Glide pose.
            return SimpleNamespace(atoms=[SimpleNamespace(name='C1',xx=1.1,xy=2.1,xz=3.1),h])
        return SimpleNamespace(atoms=[SimpleNamespace(name='N',xx=0,xy=0,xz=0),c,h])
    monkeypatch.setattr(amber_runner,'run',run)
    monkeypatch.setitem(sys.modules,'parmed',SimpleNamespace(load_file=load_file))
    ticks = iter([10.0, 35.5])
    monkeypatch.setattr(amber_runner.time, 'monotonic', lambda: next(ticks))
    amber_runner.calculate(tmp_path,{'id':1,'charge':0},'bcc',1, minimization_timeout_s=deadline, charge_timeout_s=charge_deadline)
    result = json.loads((tmp_path/'results/CMPID_1.json').read_text())
    assert result.pop('elapsed_s') == 25.5
    assert (folder/'input.mol2').read_text() == source
    assert (folder/'complex.inpcrd').read_text() == 'original coordinates'
    programs = [c[0] for c in calls]
    assert programs[:4] == ['antechamber','parmchk2','tleap','sander']
    if fail_minimization in (True, 'no_checkpoint'):
        assert 'error' in result and 'score' not in result
        assert 'MMPBSA.py' not in programs
    else:
        if deadline is not None:
            metadata = result.pop('minimization')
            assert metadata['status'] == ('time_limit_checkpoint' if fail_minimization == 'timeout' else 'completed')
            assert metadata['completed_minimization'] is (fail_minimization != 'timeout')
        assert result == {'id':1,'score':-10.25}
        assert programs[-1] == 'MMPBSA.py'


def test_existing_result_is_preserved_without_recalculation(tmp_path, monkeypatch):
    (tmp_path/'results').mkdir()
    result = tmp_path/'results/CMPID_1.json'
    result.write_text('{"id":1,"score":-20.0}\n')
    monkeypatch.setattr(amber_runner,'run',lambda *a:pytest.fail('Existing result was recalculated'))
    amber_runner.calculate(tmp_path,{'id':1,'charge':0},'bcc',1)
    assert result.read_text() == '{"id":1,"score":-20.0}\n'


@pytest.mark.parametrize('available', [True, False])
def test_timeout_uses_last_readable_finite_checkpoint_only(tmp_path, monkeypatch, available):
    def run(command, folder, lock_fd, timeout):
        assert timeout == 600
        assert 'ntwr=-10, ntxo=1' in (folder/'minimize.in').read_text()
        for step in (10, 20, 30):
            (folder/f'complex.minimized.rst7_{step}').write_text(str(step))
        raise subprocess.TimeoutExpired(command, timeout)
    def load(path, xyz=None):
        if xyz is None:
            return SimpleNamespace(atoms=[object()])
        if not available or xyz.endswith('_30'):
            raise ValueError('Incomplete checkpoint')
        return SimpleNamespace(atoms=[SimpleNamespace(xx=float('nan') if xyz.endswith('_20') else 1.,
                                                     xy=2., xz=3.)])
    monkeypatch.setattr(amber_runner, 'run', run)
    monkeypatch.setitem(sys.modules, 'parmed', SimpleNamespace(load_file=load))
    if not available:
        with pytest.raises(ValueError, match='without a valid saved checkpoint'):
            amber_runner.minimize_complex(tmp_path, 1, timeout=600)
        return
    restart = amber_runner.minimize_complex(tmp_path, 1, timeout=600)
    assert restart.read_text() == '10'
    metadata = json.loads((tmp_path/'minimization_status.json').read_text())
    assert metadata['status'] == 'time_limit_checkpoint'
    assert metadata['checkpoint_step'] == 10
    assert metadata['completed_minimization'] is False


def test_timeout_policy_does_not_rescue_native_scientific_errors(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'parmed', SimpleNamespace())
    def fail(*a, **kw):
        (tmp_path/'complex.minimized.rst7_10').write_text('checkpoint')
        raise subprocess.CalledProcessError(1, ['sander'])
    monkeypatch.setattr(amber_runner, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        amber_runner.minimize_complex(tmp_path, 1, timeout=600)
