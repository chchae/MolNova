import pytest
from rdkit import Chem

from molnova import _core


@pytest.mark.parametrize('rings', [
    ('c2ccccc2', 'C2CCCC2'),
    ('C2CCCCC2', 'C2CCCC2'),
    ('c2ccccc2', 'c2ccncc2'),
])
@pytest.mark.parametrize('reverse', [False, True])
def test_automatic_scaffold_excludes_isolated_substituent_ring_atoms(tmp_path, rings, reverse):
    # Bond-only complete-ring matching includes the first atom of each
    # terminal ring. That breaks aromatic sanitization and/or decomposition.
    smiles = [f'Clc1ccc(NC{rings[0]})cc1', f'Fc1ccc(NC{rings[1]})cc1']
    if reverse:
        smiles.reverse()
    references = [dict(id=i, name=f'elite{i}', mol=Chem.MolFromSmiles(smi))
                  for i, smi in enumerate(smiles)]

    info = _core.prepare_libinvent_scaffold(references, None, tmp_path)
    query = Chem.MolFromSmarts(info['mcs_smarts'])
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])

    # Retain the shared benzene + NC linker, with both complete terminal
    # rings left in decorators rather than copying one of their atoms.
    assert query.GetNumAtoms() == 8
    assert scaffold is not None
    assert sum(atom.IsInRing() for atom in scaffold.GetAtoms()) == 6
    assert '[*:0]' in info['scaffold_smiles']
    assert '[*:1]' in info['scaffold_smiles']
    assert all(ref['mol'].HasSubstructMatch(query) for ref in references)
    for smi in smiles:
        pair = _core.decompose_elite_for_libinvent(smi, info['scaffold_smiles'])
        assert pair is not None
        assert len(pair[1].split('|')) == 2


def test_automatic_scaffold_retains_shared_complete_substituent_ring(tmp_path):
    references = [dict(id=i, name=f'elite{i}', mol=Chem.MolFromSmiles(smi))
                  for i, smi in enumerate([
                      'Clc1ccc(NCc2ccc(Cl)cc2)cc1',
                      'Fc1ccc(NCc2ccc(F)cc2)cc1',
                  ])]
    info = _core.prepare_libinvent_scaffold(references, None, tmp_path)
    query = Chem.MolFromSmarts(info['mcs_smarts'])
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
    assert query.GetNumAtoms() == 14
    assert sum(atom.IsInRing() for atom in scaffold.GetAtoms()) == 12
    assert all(_core.decompose_elite_for_libinvent(Chem.MolToSmiles(ref['mol']),
                                                info['scaffold_smiles']) is not None
               for ref in references)


@pytest.mark.parametrize('smiles, count', [
    (['c1ccccc1', 'c1ccccc1'], 0),
    (['Clc1ccccc1', 'Fc1ccccc1'], 1),
])
def test_insufficient_attachment_sites_report_supported_recovery(tmp_path, capsys, smiles, count):
    references = [dict(id=i, name=f'elite{i}', mol=Chem.MolFromSmiles(smi))
                  for i, smi in enumerate(smiles)]

    with pytest.raises(RuntimeError, match='Fewer than two variable positions') as error:
        _core.prepare_libinvent_scaffold(references, None, tmp_path)

    # The readable core must be visible even when attachment analysis fails
    # before an iteration snapshot can be saved or logged.
    lines = capsys.readouterr().out.splitlines()
    original_index = next(i for i, line in enumerate(lines) if line.startswith('MCS SMARTS '))
    original = lines[original_index].split(': ', 1)[1]
    summary = lines[original_index + 1]
    assert summary.startswith('MCS SMARTS summary  : ')
    assert '&@' not in summary and '&!@' not in summary
    assert '&R' not in summary and '&!R' not in summary
    assert Chem.MolFromSmarts(summary.split(': ', 1)[1]) is not None
    assert '&@' in original

    message = str(error.value)
    assert f'({count} found across 2 input ligands)' in message
    assert 'elite-mcs in the project TOML' in message
    assert '[*:0] and [*:1]' in message
    assert 'fixed docking core' in message
    assert 'not reapplied in adaptive mode' in message
    assert '--mcs-smarts' not in message
    assert '--attachment-query-indices' not in message
    assert not (tmp_path / 'libinvent_scaffold.smi').exists()
    assert not (tmp_path / 'mcs.smarts').exists()
