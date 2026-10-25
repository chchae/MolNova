"""Ranked combichem MCS subsets; molecular validation does not prove affinity."""
import pytest
from rdkit import Chem
from molnova import _core
from molnova.chemistry import combichem


PREVIOUS = dict(mcs_smarts='c1ccccc1', scaffold_smiles='[*:0]c1ccc([*:1])cc1')
EXPANDED = dict(mcs_smarts='Cc1ccccc1', scaffold_smiles='[*:0]Cc1ccc([*:1])cc1')


def elites(n=10):
    return [dict(id=100+i, name=f'elite{i}', mol=Chem.MolFromSmiles('CCCc1ccc(CC)cc1')) for i in range(n)]


def test_same_core_ignores_smarts_spelling_dummy_labels_and_atom_order(monkeypatch):
    calls=[]
    def infer(selected, supplied, **kw):
        calls.append([r['id'] for r in selected])
        assert supplied is None and kw['mcs_timeout']==17
        if len(selected)==10:
            return dict(mcs_smarts='[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1',
                        scaffold_smiles='[*:1]c1ccc([*:0])cc1')
        return dict(EXPANDED)
    monkeypatch.setattr(combichem,'prepare_scaffold',infer)
    monkeypatch.setattr(combichem,'prepare_single_elite_scaffold',lambda *a:pytest.fail('Unexpected single fallback'))
    saved=combichem.prepare_ranked_elite_scaffold(elites(),PREVIOUS,'c1ccccc1',mcs_timeout=17)
    assert calls==[list(range(100,110)),list(range(100,105))]
    assert saved['source_ids']==list(range(100,105))
    assert saved['ranked_elite_ids']==list(range(100,110))
    assert saved['source']=='ranked_elite_subset_mcs'
    assert [a['status'] for a in saved['elite_mcs_attempts']]==['core_not_larger','expanded_core']


def test_first_expanded_core_uses_all_elites_without_halving(monkeypatch):
    monkeypatch.setattr(combichem,'prepare_scaffold',lambda *a,**kw:dict(EXPANDED))
    saved=combichem.prepare_ranked_elite_scaffold(elites(),PREVIOUS,'c1ccccc1')
    assert len(saved['source_ids'])==10
    assert saved['source']=='previous_iteration_elites'
    assert len(saved['elite_mcs_attempts'])==1


@pytest.mark.parametrize('n,counts', [(10,[10,5,2,1]),(7,[7,3,1]),(1,[1])])
def test_halving_reaches_one_without_computing_a_single_molecule_mcs(monkeypatch,n,counts):
    calls=[]
    def infer(selected,*a,**kw):
        assert len(selected)>1
        calls.append(len(selected))
        return dict(PREVIOUS)
    monkeypatch.setattr(combichem,'prepare_scaffold',infer)
    monkeypatch.setattr(combichem.secrets,'choice',lambda choices:choices[-1])
    saved=combichem.prepare_ranked_elite_scaffold(elites(n),PREVIOUS,'c1ccccc1')
    assert calls==counts[:-1]
    assert [a['elite_count'] for a in saved['elite_mcs_attempts']]==counts
    assert saved['source']=='single_elite_sidechain_replacement'
    assert saved['source_ids']==[100]
    assert saved['replacement_candidates'] and saved['replacement_cuts']


@pytest.mark.parametrize('candidate,reason', [
    (dict(mcs_smarts='C1CCCCC1C',scaffold_smiles='[*:0]C1CCCCC1C[*:1]'),'previous_core_not_retained'),
    (dict(mcs_smarts='CCCc1ccccc1',scaffold_smiles='CCCc1ccccc1[*:0]'),'docking_core_not_retained'),
])
def test_invalid_growth_halves_instead_of_replacing_previous_or_docking_core(monkeypatch,candidate,reason):
    monkeypatch.setattr(combichem,'prepare_scaffold',lambda selected,*a,**kw:
                        dict(candidate if len(selected)==10 else EXPANDED))
    docking='[*]c1ccc([*])cc1' if reason=='docking_core_not_retained' else 'c1ccccc1'
    saved=combichem.prepare_ranked_elite_scaffold(elites(),PREVIOUS,docking)
    assert saved['elite_mcs_attempts'][0]['status']==reason
    assert saved['source_ids']==list(range(100,105))


def test_every_elite_in_the_accepted_subset_must_decompose(monkeypatch):
    refs=elites()
    refs[4]['mol']=Chem.MolFromSmiles('Cc1ccc(C)cc1')  # too short for EXPANDED's dummy
    monkeypatch.setattr(combichem,'prepare_scaffold',lambda *a,**kw:dict(EXPANDED))
    saved=combichem.prepare_ranked_elite_scaffold(refs,PREVIOUS,'c1ccccc1')
    assert [a['elite_count'] for a in saved['elite_mcs_attempts']]==[10,5,2]
    assert [a['status'] for a in saved['elite_mcs_attempts']]==[
        'elite_decomposition_failed','elite_decomposition_failed','expanded_core']
    assert saved['source_ids']==[100,101]


def test_boundary_failure_is_recorded_before_trying_smaller_prefix(monkeypatch):
    def infer(selected,*a,**kw):
        if len(selected)==10:
            raise combichem.NonSingleAttachmentBoundaryError('C=O')
        return dict(EXPANDED)
    monkeypatch.setattr(combichem,'prepare_scaffold',infer)
    saved=combichem.prepare_ranked_elite_scaffold(elites(),PREVIOUS,'c1ccccc1')
    assert saved['elite_mcs_attempts'][0]['mcs_smarts']=='C=O'
    assert saved['elite_mcs_attempts'][0]['status']=='non_single_attachment_boundary'
    assert saved['elite_mcs_attempts'][1]['status']=='expanded_core'


@pytest.mark.parametrize('message', ['RDKit returned an invalid/empty MCS SMARTS.',
                                   'RDKit MCS search timed out'])
def test_only_typed_search_cancellation_is_recoverable(monkeypatch, message):
    calls = []
    def fail(selected, *args, **kwargs):
        calls.append(len(selected))
        raise RuntimeError(message)
    monkeypatch.setattr(combichem, 'prepare_scaffold', fail)
    with pytest.raises(RuntimeError, match=message):
        combichem.prepare_ranked_elite_scaffold(elites(), PREVIOUS, 'c1ccccc1')
    assert calls == [10]


def test_timeout_after_rejected_core_identifies_the_failing_subset(monkeypatch, capsys):
    calls = []
    def infer(selected, *args, **kwargs):
        calls.append(len(selected))
        assert kwargs['mcs_timeout'] == 600
        if len(selected) == 10:
            return dict(PREVIOUS)
        if len(selected) == 5:
            raise _core.MCSSearchTimeoutError('RDKit MCS search timed out')
        return dict(EXPANDED)
    monkeypatch.setattr(combichem, 'prepare_scaffold', infer)
    saved = combichem.prepare_ranked_elite_scaffold(elites(), PREVIOUS, 'c1ccccc1', mcs_timeout=600)
    assert calls == [10, 5, 2]
    assert saved['elite_mcs_policy'] == 'ranked_halving_v2'
    assert saved['source_ids'] == [100, 101]
    assert saved['elite_mcs_attempts'][1] == dict(
        elite_count=5, source_ids=list(range(100, 105)), status='mcs_timeout',
        mcs_smarts=None, timeout_seconds=600, failure_message='RDKit MCS search timed out')
    output = capsys.readouterr().out
    assert '10 elites: core_not_larger; trying highest-ranked 5' in output
    assert 'Trying highest-ranked 5/10 elites (timeout=600s)' in output
    assert '5 elites: mcs_timeout; trying highest-ranked 2' in output
    assert 'Trying highest-ranked 2/10' in output


@pytest.mark.parametrize('all_timeouts', [False, True])
def test_two_elite_timeout_uses_real_single_elite_replacement(monkeypatch, all_timeouts):
    calls = []
    def infer(selected, *args, **kwargs):
        calls.append(len(selected))
        if all_timeouts or len(selected) == 2:
            raise _core.MCSSearchTimeoutError('RDKit MCS search timed out')
        return dict(PREVIOUS)
    monkeypatch.setattr(combichem, 'prepare_scaffold', infer)
    monkeypatch.setattr(combichem.secrets, 'choice', lambda choices: choices[-1])
    saved = combichem.prepare_ranked_elite_scaffold(elites(), PREVIOUS, 'c1ccccc1')
    assert calls == [10, 5, 2]
    assert [a['elite_count'] for a in saved['elite_mcs_attempts']] == [10, 5, 2, 1]
    assert saved['elite_mcs_attempts'][-2]['status'] == 'mcs_timeout'
    assert saved['source'] == 'single_elite_sidechain_replacement'
    assert saved['source_ids'] == [100]
    assert saved['replacement_cuts'] and saved['replacement_candidates']
    assert all(a['mcs_smarts'] is None for a in saved['elite_mcs_attempts'] if a['status'] == 'mcs_timeout')


def test_canceled_mcs_rejects_partial_result_and_reports_supported_configuration(monkeypatch):
    from types import SimpleNamespace
    def find(mols, parameters):
        assert parameters.Timeout == 600
        return SimpleNamespace(canceled=True, smartsString='c1ccccc1')
    monkeypatch.setattr(_core.rdFMCS, 'FindMCS', find)
    with pytest.raises(_core.MCSSearchTimeoutError, match='timed out after 600 seconds for 5 ligands') as error:
        _core.infer_mcs_smarts_from_iteration0(elites(5), timeout=600)
    assert 'mcs-timeout in the project TOML' in str(error.value)
    assert '--mcs-timeout' not in str(error.value)
    assert '--mcs-smarts' not in str(error.value)


def test_missing_or_incompatible_history_stops(monkeypatch):
    with pytest.raises(RuntimeError,match='preceding scaffold'):
        combichem.prepare_ranked_elite_scaffold(elites(),None,'c1ccccc1')
    with pytest.raises(RuntimeError,match='fixed docking core'):
        combichem.prepare_ranked_elite_scaffold(elites(),PREVIOUS,'C1CCCCC1')


def test_real_mcs_search_expands_after_halving_the_ranked_set():
    smiles=['CCCc1ccc(CCO)cc1','CCCc1ccc(CCN)cc1','CCCc1ccc(CCF)cc1',
            'CCCc1ccc(CCCl)cc1','CCCc1ccc(CCBr)cc1',
            'Nc1ccc(O)cc1','CNc1ccc(OC)cc1','CCNc1ccc(OC)cc1',
            'CNc1ccc(OCC)cc1','CCNc1ccc(OCC)cc1']
    ranked=[dict(id=i,name=f'elite{i}',mol=Chem.MolFromSmiles(s)) for i,s in enumerate(smiles)]
    saved=combichem.prepare_ranked_elite_scaffold(ranked,PREVIOUS,'c1ccccc1')
    assert [a['elite_count'] for a in saved['elite_mcs_attempts']]==[10,5]
    assert saved['elite_mcs_attempts'][0]['status']=='core_not_larger'
    assert saved['elite_mcs_attempts'][1]['core_heavy_atoms']>6
    assert saved['source_ids']==[0,1,2,3,4]
    assert all(_core.decompose_elite_for_libinvent(smiles[i],saved['scaffold_smiles']) is not None
               for i in range(5))
