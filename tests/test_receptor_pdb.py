import fcntl
import json
from pathlib import Path

import pytest

from molnova import amber_runner
from molnova.receptor_pdb import NORMALIZATION_VERSION, normalize_receptor_pdb


FIXTURE = Path(__file__).parent / 'data/amber_receptor_names.pdb'


def normalize(tmp_path, text):
    source = tmp_path / 'receptor.pdb'
    source.write_bytes(text.encode('ascii'))
    destination = tmp_path / 'receptor.normalized.pdb'
    changes = normalize_receptor_pdb(source, destination, tmp_path / 'calculation.log')
    assert source.read_bytes() == text.encode('ascii')
    return destination.read_text(), changes


def atoms(text):
    return [line for line in text.splitlines() if line[:6].strip() in ('ATOM', 'HETATM')]


def residue_lines(resname):
    return ''.join(line for line in FIXTURE.read_text().splitlines(keepends=True)
                   if line[:6].strip() in ('ATOM', 'TER') and line[17:20] == resname)


def rename_field(text, start, end, before, after):
    return ''.join(line[:start] + after + line[end:] if line[start:end] == before else line
                   for line in text.splitlines(keepends=True))


def test_failed_receptor_names_identifiers_coordinates_and_idempotence(tmp_path):
    original = FIXTURE.read_bytes()
    source = tmp_path / 'source.pdb'
    source.write_bytes(original)
    result = tmp_path / 'normalized.pdb'
    log = tmp_path / 'calculation.log'
    changes = normalize_receptor_pdb(source, result, log)
    before = original.decode().splitlines(keepends=True)
    after = result.read_bytes().decode().splitlines(keepends=True)
    assert len(before) == len(after)
    for a, b in zip(before, after):
        if a[:6].strip() in ('ATOM', 'HETATM', 'TER'):
            assert a[:12] == b[:12]
            assert a[16:17] == b[16:17]
            assert a[20:] == b[20:]  # chain, resnum, insertion, xyz, occupancy, B, element, charge
        else:
            assert a == b
    residues = {}
    for line in atoms(result.read_text()):
        residues.setdefault(line[21:27], set()).add((line[17:20], line[12:16].strip()))
    for number in (773, 805):
        assert {name for name, _ in residues[f'A{number:4d} ']} == {'HID'}
    for number in (835, 850, 870, 888, 893):
        assert {name for name, _ in residues[f'A{number:4d} ']} == {'HIE'}
    assert residues['A 981A'] == {('NME', name) for name in ('N','H','CH3','HH31','HH32','HH33')}
    assert {atom for _, atom in residues['A 696 ']} == {'CH3','C','O','HH31','HH32','HH33'}
    assert ('GLU','H') in residues['A 697 '] and ('GLU','H1') not in residues['A 697 ']
    assert any(line.startswith('TER') and line[17:27] == 'NME A 981A' for line in after)
    assert source.read_bytes() == original
    assert len(changes) == 16 and all(change in log.read_text() for change in changes)
    second = tmp_path / 'second.pdb'
    assert normalize_receptor_pdb(result, second, log) == []
    assert second.read_bytes() == result.read_bytes()


@pytest.mark.parametrize('ring,expected', [('HD1','HID'), ('HE2','HIE'), ('both','HIP')])
def test_histidine_protonation_and_explicit_states(tmp_path, ring, expected):
    original = residue_lines('HIS')
    # Single residue, retain its complete heavy atoms and ring C hydrogens.
    original = ''.join(line for line in original.splitlines(keepends=True) if line[22:26] == ' 773')
    if ring != 'HD1':
        original = rename_field(original, 12, 16, ' HD1', ' HE2')
    if ring == 'both':
        extra = next(line for line in original.splitlines(keepends=True) if line[12:16] == ' HE2')
        original += extra[:12] + ' HD1' + extra[16:]
    result, _ = normalize(tmp_path, original)
    assert all(line[17:20] == expected for line in atoms(result))
    assert normalize(tmp_path, result)[1] == []


@pytest.mark.parametrize('invalid', ['no_ring_h', 'explicit_conflict', 'unexpected_cap',
                                   'missing_cap_h', 'alias_collision', 'cap_geometry',
                                   'alternate_atom', 'duplicate_atom', 'truncated', 'no_atoms', 'models',
                                   'element', 'nonfinite', 'backbone_collision'])
def test_invalid_inputs_fail_without_publishing(tmp_path, invalid):
    text = FIXTURE.read_text()
    if invalid == 'no_ring_h':
        text = ''.join(line for line in text.splitlines(keepends=True)
                       if not (line[17:20] == 'HIS' and line[12:16].strip() in ('HD1','HE2')))
    elif invalid == 'explicit_conflict':
        text = rename_field(text, 17, 20, 'HIS', 'HIE')
    elif invalid == 'unexpected_cap':
        text = rename_field(text, 12, 16, '1HA ', ' XX ')
    elif invalid == 'missing_cap_h':
        text = ''.join(line for line in text.splitlines(keepends=True) if line[12:16] != '1HA ')
    elif invalid == 'alias_collision':
        text += next(line for line in text.splitlines(keepends=True) if line[12:16] == '1HA ').replace('1HA ', 'HH31', 1)
    elif invalid == 'cap_geometry':
        text = ''.join(
            line[:30] + ' 999.000' + line[38:] if line[17:20] == 'ACE' and line[12:16].strip() == 'CH3' else line
            for line in text.splitlines(keepends=True))
    elif invalid == 'element':
        text = ''.join(line[:76] + ' N' + line[78:] if line[17:20] == 'ACE' and line[12:16].strip() == 'CH3' else line
                       for line in text.splitlines(keepends=True))
    elif invalid == 'nonfinite':
        text = ''.join(line[:30] + '     nan' + line[38:] if line[17:20] == 'ACE' else line
                       for line in text.splitlines(keepends=True))
    elif invalid == 'backbone_collision':
        extra = next(line for line in text.splitlines(keepends=True) if line[12:16].strip() == 'H1')
        text = ''.join(line + (extra[:12] + ' H  ' + extra[16:] if line == extra else '')
                       for line in text.splitlines(keepends=True))
    elif invalid == 'alternate_atom':
        text = rename_field(text, 16, 17, ' ', 'A')
    elif invalid == 'duplicate_atom':
        text = text + atoms(text)[0] + '\n'
    elif invalid == 'truncated':
        text = 'ATOM      1  N   GLU A 697\n'
    elif invalid == 'no_atoms':
        text = 'END\n'
    elif invalid == 'models':
        text = 'MODEL        1\n' + text + 'ENDMDL\nMODEL        2\n' + text
    source = tmp_path / 'raw.pdb'
    source.write_text(text)
    target = tmp_path / 'normalized.pdb'
    log = tmp_path / 'calculation.log'
    with pytest.raises(ValueError):
        normalize_receptor_pdb(source, target, log)
    assert not target.exists()
    assert 'normalization failed' in log.read_text()
    assert source.read_text() == text


@pytest.mark.parametrize('boundary', ['ter', 'chain', 'distance', 'unrelated'])
def test_h1_is_only_changed_for_connected_ace(tmp_path, boundary):
    ace = residue_lines('ACE')
    glu = residue_lines('GLU')
    if boundary == 'ter':
        text = ace + 'TER       7      ACE A 696 \n' + glu
    elif boundary == 'chain':
        text = ace + rename_field(glu, 21, 22, 'A', 'B')
    elif boundary == 'distance':
        text = ace + ''.join(line[:30] + ' 999.000' + line[38:] if line[12:16].strip() == 'N' else line
                             for line in glu.splitlines(keepends=True))
    else:
        text = glu
    result, _ = normalize(tmp_path, text)
    assert any(line[12:16].strip() == 'H1' for line in atoms(result))
    assert result.count('TER') == text.count('TER')


def test_partial_cap_aliases_and_crlf_are_idempotent(tmp_path):
    text = residue_lines('NMA')
    text = rename_field(text, 12, 16, '1HA ', 'HH31')
    text = rename_field(text, 12, 16, '2HA ', ' HA2')
    text = text.replace('\n', '\r\n')
    raw = tmp_path / 'raw.pdb'
    raw.write_bytes(text.encode())
    normalized = tmp_path / 'normalized.pdb'
    normalize_receptor_pdb(raw, normalized, tmp_path / 'calculation.log')
    assert normalized.read_bytes().count(b'\r\n') == text.count('\r\n')
    second = tmp_path / 'second.pdb'
    assert normalize_receptor_pdb(normalized, second, tmp_path / 'calculation.log') == []
    assert second.read_bytes() == normalized.read_bytes()


@pytest.mark.parametrize('failure', ['fatal', 'summary', 'missing', 'empty'])
def test_leap_zero_exit_does_not_allow_invalid_topology(tmp_path, monkeypatch, failure):
    products = ('receptor.prmtop', 'receptor.inpcrd')
    for name in products:
        (tmp_path / name).write_text('stale topology')
    def run(command, folder, lock_fd):
        assert all(not (folder / name).exists() for name in products)
        for name in products:
            if failure != 'missing':
                (folder / name).write_text('' if failure == 'empty' else 'new topology')
        (folder / 'calculation.log').write_text(
            'FATAL: Atom has no type\n' if failure == 'fatal' else
            'Exiting LEaP: Errors = 2; Warnings = 0\n' if failure == 'summary' else '')
    monkeypatch.setattr(amber_runner, 'run', run)
    with pytest.raises(RuntimeError):
        amber_runner.run_leap('receptor_leap.in', tmp_path, 1, products)


def test_worker_stops_batch_on_bad_receptor_before_leap_or_ligands(tmp_path, monkeypatch):
    (tmp_path / 'input.json').write_text(json.dumps({'cpus':1, 'entries':[{'id':1}],
        'charge_method':'bcc', 'receptor_normalization_version':NORMALIZATION_VERSION}))
    (tmp_path / 'receptor.pdb').write_text('END\n')
    monkeypatch.setattr(amber_runner, 'run', lambda *a:pytest.fail('LEaP started with invalid receptor'))
    monkeypatch.setattr(amber_runner, 'calculate', lambda *a:pytest.fail('Ligand calculation started'))
    amber_runner.execute(tmp_path)
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'failed'
    assert 'No receptor atoms' in (tmp_path / 'calculation.log').read_text()


def test_worker_lock_preserves_live_topology(tmp_path):
    topology = tmp_path / 'receptor.prmtop'
    topology.write_text('live topology')
    with (tmp_path / 'run.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        amber_runner.execute(tmp_path)
    assert topology.read_text() == 'live topology'


@pytest.mark.parametrize('bad_topology', [False, True])
def test_worker_uses_normalized_receptor_and_parses_before_ligands(tmp_path, monkeypatch, bad_topology):
    import sys
    from types import SimpleNamespace
    original = FIXTURE.read_bytes()
    (tmp_path / 'receptor.pdb').write_bytes(original)
    (tmp_path / 'receptor_bonds.in').write_text('')
    (tmp_path / 'input.json').write_text(json.dumps({'cpus':1, 'entries':[{'id':1}],
        'charge_method':'bcc', 'receptor_normalization_version':NORMALIZATION_VERSION}))
    def run(command, folder, lock_fd):
        script = (folder / 'receptor_leap.in').read_text()
        assert 'rec = loadpdb receptor.normalized.pdb' in script
        assert 'set NME.1.C name CH3' in script
        assert 'set ACE.1.H1 name HH31' in script
        assert b'NME A 981A' in (folder / 'receptor.normalized.pdb').read_bytes()
        assert (folder / 'receptor.pdb').read_bytes() == original
        with (folder / 'calculation.log').open('a') as log:
            log.write('Exiting LEaP: Errors = 0; Warnings = 0\n')
        (folder / 'receptor.prmtop').write_text('topology')
        (folder / 'receptor.inpcrd').write_text('coordinates')
    parsed = []
    def load_file(*args, **kwargs):
        if bad_topology:
            raise ValueError('Unreadable topology')
        parsed.append(args)
    calculated = []
    def calculate(*args):
        assert parsed
        calculated.append(args)
    monkeypatch.setitem(sys.modules, 'parmed', SimpleNamespace(load_file=load_file))
    monkeypatch.setattr(amber_runner, 'run', run)
    monkeypatch.setattr(amber_runner, 'calculate', calculate)
    amber_runner.execute(tmp_path)
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == ('failed' if bad_topology else 'done')
    assert len(calculated) == (0 if bad_topology else 1)


def test_normalizer_refuses_in_place_modification(tmp_path):
    source = tmp_path / 'receptor.pdb'
    source.write_bytes(FIXTURE.read_bytes())
    with pytest.raises(ValueError, match='separate output'):
        normalize_receptor_pdb(source, source, tmp_path / 'calculation.log')
    assert source.read_bytes() == FIXTURE.read_bytes()
