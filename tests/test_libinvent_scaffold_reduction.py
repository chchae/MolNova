import json

import pytest
from rdkit import Chem

from molnova import _core, database
from molnova.chemistry.combichem import assemble
from molnova.chemistry.libinvent_scaffold import reduce_to_two_attachments
from molnova.chemistry.scaffold import prepare_iteration_scaffold
from test_iteration_scaffold import project


PREVIOUS = '[*:0]c1ccc(N[*:1])cc1'
INFERRED = 'Cc1ccc(NCCCC)cc1'
BRANCHED = [
    'CCc1ccc(NCCC(CN)c2ccccc2)cc1',
    'OCc1ccc(NCCC(CO)c2ccncc2)cc1',
]
EGFR_PREVIOUS = '[*:0]Cc1ccc(-c2cc3c(N[*:1])ncnc3[nH]2)cc1'
EGFR_ELITES = [
    'CCc1ccc(-c2cc3c(NC(CN4CCN(CC)CC4)c4ccncc4)ncnc3[nH]2)cc1',
    'CN(C)Cc1ccc(-c2cc3c(NC(CN4CCN(C)CC4)c4ccnnc4)ncnc3[nH]2)cc1',
    'CCc1ccc(-c2cc3c(NC(CN4CCN(C)CC4)c4ccns4)ncnc3[nH]2)cc1',
    'CCc1ccc(-c2cc3c(NC(CN4CCN(CC)CC4)c4cccnc4)ncnc3[nH]2)cc1',
    'CCc1ccc(-c2cc3c(NC(CN4CCN(CC)CC4)c4ccccn4)ncnc3[nH]2)cc1',
    'CN(C)Cc1ccc(-c2cc3c(NC(CN4CCN(C)CC4)c4ccnn4C)ncnc3[nH]2)cc1',
    'CCc1ccc(-c2cc3c(NC(CN4CCN(C)CC4)c4ccnnc4)ncnc3[nH]2)cc1',
    'CCc1ccc(-c2cc3c(NC(CCN4CCN(C)CC4)c4cccnc4)ncnc3[nH]2)cc1',
    'CN(C)Cc1ccc(-c2cc3c(NC(CN4CCN(C)CC4)c4ccno4)ncnc3[nH]2)cc1',
    'CCc1ccc(-c2cc3c(NC(CN4CCN(CC)CC4)c4cncs4)ncnc3[nH]2)cc1',
]


def inputs(smiles):
    return [dict(id=i, name=f'E{i}', mol=Chem.MolFromSmiles(s)) for i, s in enumerate(smiles)]


def assert_round_trips(smiles, scaffold):
    core = Chem.MolFromSmiles(scaffold)
    assert sorted(a.GetAtomMapNum() for a in core.GetAtoms() if a.GetAtomicNum() == 0) == [0, 1]
    for s in smiles:
        pair = _core.decompose_elite_for_libinvent(s, scaffold)
        assert pair is not None
        assert assemble(core, [Chem.MolFromSmiles(d) for d in pair[1].split('|')]) == Chem.MolToSmiles(Chem.MolFromSmiles(s))


def test_linear_extensions_stop_before_branches_and_reassemble_all_elites():
    previous = dict(scaffold_smiles=PREVIOUS)
    result = reduce_to_two_attachments(inputs(BRANCHED), INFERRED, previous, 'c1ccccc1')
    core = Chem.MolFromSmiles(result['scaffold_smiles'])
    # Keep benzene, N, one benzyl C and two N-side methylenes, stopping
    # before the branched carbon rather than silently omitting its bonds.
    assert core.GetNumHeavyAtoms() == 10
    assert core.HasSubstructMatch(Chem.MolFromSmarts('c1ccc(NCC)cc1'))
    assert core.HasSubstructMatch(Chem.MolFromSmarts(result['mcs_smarts']))
    assert previous == dict(scaffold_smiles=PREVIOUS)
    assert_round_trips(BRANCHED, result['scaffold_smiles'])
    assert reduce_to_two_attachments(inputs(BRANCHED), INFERRED, previous, 'c1ccccc1') == result


def test_candidate_incompatible_with_one_elite_uses_previous_without_dropping_it():
    smiles = [BRANCHED[0], 'OCc1ccc(NC(C)CC(CO)c2ccncc2)cc1']
    assert_round_trips(smiles, PREVIOUS)
    assert reduce_to_two_attachments(inputs(smiles), INFERRED,
                                     dict(scaffold_smiles=PREVIOUS), 'c1ccccc1') is None


def test_missing_previous_core_embedding_does_not_invent_a_smaller_core():
    assert reduce_to_two_attachments(inputs(BRANCHED), 'c1ccccc1',
                                     dict(scaffold_smiles=PREVIOUS), 'c1ccccc1') is None


def test_previous_must_reassemble_all_selected_elites():
    with pytest.raises(RuntimeError, match='exactly reassemble'):
        reduce_to_two_attachments(inputs([BRANCHED[0], 'Cc1ccc(Cl)cc1']), INFERRED,
                                  dict(scaffold_smiles=PREVIOUS), 'c1ccccc1')


def test_two_external_bonds_at_one_core_atom_count_as_two_boundaries():
    references = inputs(['CCc1ccc(NC(CN)CO)cc1', 'OCc1ccc(NC(CO)CN)cc1'])
    with pytest.raises(_core.ExcessAttachmentSitesError) as error:
        _core.determine_libinvent_attachment_sites(
            references, 'Cc1ccc(NC)cc1', reject_excess=True,
        )
    assert error.value.count == 3


def test_linear_extension_preserves_isotopes_and_charged_core():
    smiles = [s.replace('NC', '[NH2+][13CH2]', 1) for s in BRANCHED]
    previous = PREVIOUS.replace('N', '[NH2+]')
    result = reduce_to_two_attachments(inputs(smiles), 'Cc1ccc([#7]CCCC)cc1',
                                       dict(scaffold_smiles=previous), 'c1ccccc1')
    assert result is not None
    assert '[13CH2]' in result['scaffold_smiles']
    assert '[NH2+]' in result['scaffold_smiles']
    assert_round_trips(smiles, result['scaffold_smiles'])


def test_previous_attachment_labels_survive_reversed_smiles_order():
    previous = '[*:1]c1ccc(N[*:0])cc1'
    result = reduce_to_two_attachments(inputs(BRANCHED), INFERRED,
                                       dict(scaffold_smiles=previous), 'c1ccccc1')
    core = Chem.MolFromSmiles(result['scaffold_smiles'])
    nitrogen = next(a.GetIdx() for a in core.GetAtoms() if a.GetAtomicNum() == 7)
    distances = {a.GetAtomMapNum(): len(Chem.GetShortestPath(core, nitrogen, a.GetIdx())) - 1
                 for a in core.GetAtoms() if a.GetAtomicNum() == 0}
    assert distances[0] == 3  # Two retained N-side methylenes, then its original site 0.
    assert distances[1] > distances[0]
    assert_round_trips(BRANCHED, result['scaffold_smiles'])


def seed(project, smiles, previous=PREVIOUS):
    docking = 'c1ccccc1'
    database.iteration_scaffold(project.db_path, 0, dict(mcs_smarts=docking,
                                source='user_initial_mcs', initial_elite_mcs=None))
    # Retain the actual concrete previous core, without its dummies.
    from molnova.chemistry.combichem import _retained_core_query
    core = Chem.MolToSmarts(_retained_core_query(Chem.MolFromSmiles(previous)))
    saved = dict(mcs_smarts=core, scaffold_smiles=previous, source='previous_iteration_elites', source_ids=[])
    database.iteration_scaffold(project.db_path, 1, saved)
    project.prime_gbsa_elite_count = len(smiles)
    elites = [(i, f'E{i}', s, 1, -10, -80 + i) for i, s in enumerate(smiles)]
    return elites, saved


def test_new_iteration_adjusts_and_resume_keeps_choice(project, monkeypatch, capsys):
    elites, previous = seed(project, BRANCHED)
    monkeypatch.setattr(_core, 'infer_mcs_smarts_from_iteration0', lambda *a, **k: INFERRED)
    prepare_iteration_scaffold(project, 2, elites, create=True)
    saved = database.iteration_scaffold(project.db_path, 2)
    assert saved['source'] == 'previous_attachment_guided_mcs'
    assert saved['attempted_attachment_count'] == 3
    assert saved['attachment_from_iteration'] == 1
    assert saved['source_ids'] == [0, 1]
    assert saved['attempted_mcs_smarts'] == INFERRED
    assert saved['docking_mcs_smarts'] == 'c1ccccc1'
    assert database.iteration_scaffold(project.db_path, 1) == previous
    assert_round_trips(BRANCHED, saved['scaffold_smiles'])
    assert 'two-site MCS adjustment' in capsys.readouterr().out
    monkeypatch.setattr(_core, 'prepare_libinvent_scaffold', lambda *a, **k: pytest.fail('recomputed'))
    prepare_iteration_scaffold(project, 2, create=True)
    assert database.iteration_scaffold(project.db_path, 2) == saved
    assert json.loads((project.output / 'iter02/01-scaffold/scaffold_info.json').read_text()) == saved


def test_real_iteration8_elites_reuse_previous_before_the_three_way_branch(project, capsys):
    elites, previous = seed(project, EGFR_ELITES, EGFR_PREVIOUS)
    prepare_iteration_scaffold(project, 2, elites, create=True)
    saved = database.iteration_scaffold(project.db_path, 2)
    assert saved['source'] == 'previous_iteration_scaffold_fallback'
    assert saved['fallback_reason'] == 'excess_attachment_sites'
    assert saved['scaffold_smiles'] == EGFR_PREVIOUS
    assert saved['mcs_smarts'] == previous['mcs_smarts']
    assert 'exceeded two attachment boundaries' in capsys.readouterr().out
    assert_round_trips(EGFR_ELITES, saved['scaffold_smiles'])


def test_supplied_core_is_not_automatically_reduced(project, monkeypatch):
    elites, _ = seed(project, BRANCHED)
    project.elite_mcs_smarts = INFERRED
    monkeypatch.setattr('molnova.chemistry.libinvent_scaffold.reduce_to_two_attachments',
                        lambda *a: pytest.fail('supplied core changed'))
    prepare_iteration_scaffold(project, 2, elites, create=True)
    assert database.iteration_scaffold(project.db_path, 2)['source'] == 'user_elite_mcs'


@pytest.mark.parametrize('problem', ['missing_previous', 'incompatible_docking_core'])
def test_excess_boundaries_fail_closed_without_valid_previous_scaffold(project, monkeypatch, problem):
    elites, _ = seed(project, BRANCHED)
    with database.connect(project.db_path) as conn:
        if problem == 'missing_previous':
            conn.execute('DELETE FROM iteration_scaffold WHERE iteration=1')
        else:
            saved = json.loads(conn.execute('SELECT snapshot FROM iteration_scaffold WHERE iteration=0').fetchone()[0])
            saved['mcs_smarts'] = 'C1CCCCC1'
            conn.execute('UPDATE iteration_scaffold SET snapshot=? WHERE iteration=0', (json.dumps(saved),))
            conn.execute("UPDATE compound SET smiles='CC1CCCCC1' WHERE name='R3'")
        before = conn.execute('SELECT * FROM compound ORDER BY id').fetchall()
    monkeypatch.setattr(_core, 'infer_mcs_smarts_from_iteration0', lambda *a, **k: INFERRED)
    with pytest.raises(RuntimeError, match='no saved scaffold|fixed docking core'):
        prepare_iteration_scaffold(project, 2, elites, create=True)
    assert database.iteration_scaffold(project.db_path, 2) is None
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT * FROM compound ORDER BY id').fetchall() == before
