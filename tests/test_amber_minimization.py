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


@pytest.mark.parametrize('fail_minimization', [False, True])
def test_calculate_scores_minimized_complex_and_never_scores_failed_minimization(tmp_path, monkeypatch, fail_minimization):
    folder = tmp_path/'CMPID_1'
    folder.mkdir()
    (tmp_path/'results').mkdir()
    (tmp_path/'receptor_bonds.in').write_text('')
    source = '@<TRIPOS>ATOM\n1 C1 1.0 2.0 3.0 C.3 1 LIG 0\n2 H2 4.0 5.0 6.0 H 1 LIG 0\n@<TRIPOS>BOND\n1 1 2 1\n'
    (folder/'input.mol2').write_text(source)
    calls = []
    def run(command, folder, lock_fd):
        calls.append(command)
        if command[0] == 'antechamber':
            (folder/'ligand.mol2').write_text(source)
        elif command[0] == 'tleap':
            assert 'rec = loadpdb ../receptor.normalized.pdb' in (folder/'leap.in').read_text()
            (folder/'calculation.log').write_text('Exiting LEaP: Errors = 0\n')
            for name in ('complex','ligand','receptor'):
                (folder/f'{name}.prmtop').write_text('topology')
                (folder/f'{name}.inpcrd').write_text('original coordinates')
        elif command[0] == 'sander':
            (folder/'minimize.out').write_text('partial output' if fail_minimization else FINAL)
            (folder/'complex.minimized.rst7').write_text('relaxed coordinates')
        elif command[0] == 'MMPBSA.py':
            assert command[command.index('-y')+1] == 'complex.minimized.rst7'
            assert 'restraint' not in (folder/'mmpbsa.in').read_text()
            (folder/'FINAL_RESULTS_MMPBSA.dat').write_text('DELTA TOTAL -10.25\n')
    c = SimpleNamespace(name='C1',xx=1,xy=2,xz=3)
    h = SimpleNamespace(name='H2',xx=4,xy=5,xz=6)
    def load_file(path, xyz):
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
    amber_runner.calculate(tmp_path,{'id':1,'charge':0},'bcc',1)
    result = json.loads((tmp_path/'results/CMPID_1.json').read_text())
    assert result.pop('elapsed_s') == 25.5
    assert (folder/'input.mol2').read_text() == source
    assert (folder/'complex.inpcrd').read_text() == 'original coordinates'
    programs = [c[0] for c in calls]
    assert programs[:4] == ['antechamber','parmchk2','tleap','sander']
    if fail_minimization:
        assert 'error' in result and 'score' not in result
        assert 'MMPBSA.py' not in programs
    else:
        assert result == {'id':1,'score':-10.25}
        assert programs[-1] == 'MMPBSA.py'


def test_existing_result_is_preserved_without_recalculation(tmp_path, monkeypatch):
    (tmp_path/'results').mkdir()
    result = tmp_path/'results/CMPID_1.json'
    result.write_text('{"id":1,"score":-20.0}\n')
    monkeypatch.setattr(amber_runner,'run',lambda *a:pytest.fail('Existing result was recalculated'))
    amber_runner.calculate(tmp_path,{'id':1,'charge':0},'bcc',1)
    assert result.read_text() == '{"id":1,"score":-20.0}\n'
