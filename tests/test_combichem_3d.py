"""Artificial-coordinate regressions, not binding-site/affinity validation."""
import copy
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from molnova import database
from molnova.config import combichem_3d_settings
from molnova.states import PoseVerdict
from molnova.chemistry.combichem3d import history as provenance
from molnova.chemistry.combichem3d import search as conformations
from molnova.chemistry.combichem3d.assembly import assemble, growth_sites, hydrogen_directions
from molnova.chemistry.combichem3d.clashes import BumpChecker, load_receptor
from molnova.chemistry.combichem3d.library import load_library, normalize, candidate_pools, combination_at
from molnova.chemistry.combichem3d.evaluation import Evaluator
from molnova.stages import combichem_3d as stage


def toy(directory, iteration=1, attachments=1):
    path = Path(__file__).resolve().parents[1]/'examples/combichem-3d/make_toy_inputs.py'
    spec = importlib.util.spec_from_file_location('molnova_toy_input',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    filename = module.create(directory,iteration,attachments)
    return filename,combichem_3d_settings(filename)


@pytest.fixture
def inputs(tmp_path):
    filename,cfg = toy(tmp_path/'toy')
    cfg.update({'conformers':1,'torsion-trials':0,'optimization-iterations':1000,'processes':1})
    mol,history = provenance.load_parents(cfg['parents'],cfg['history'])[0]
    return filename,cfg,mol,history


def parent(smiles, core=(0,)):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol,randomSeed=37) >= 0
    AllChem.UFFOptimizeMolecule(mol,maxIters=1000)
    mol = Chem.RemoveHs(mol)
    history = dict(schema=provenance.SCHEMA,iteration=1,
        core_smarts=Chem.MolFragmentToSmarts(mol,list(core)), attachments=[f'a{core[0]}'],
        core=[dict(core_index=j,uid=f'a{i}',xyz=list(mol.GetConformer().GetAtomPosition(i))) for j,i in enumerate(core)],
        atoms=[dict(index=i,uid=f'a{i}',branch='core' if i in core else 'branch_1',
                    fragment='core' if i in core else 'first',added_iteration=0 if i in core else 1)
               for i in range(mol.GetNumAtoms())])
    return provenance.annotate(mol,history)


@pytest.mark.parametrize('count',[1,2,3,4])
def test_iteration_one_uses_every_core_attachment_and_full_cartesian_space(tmp_path,count):
    _,cfg = toy(tmp_path/f'toy{count}',0,count)
    cfg['sidechains-per-site'] = 2
    mol,h = provenance.load_parents(cfg['parents'],cfg['history'])[0]
    sites = growth_sites(mol,h)
    assert sites == h['attachments'] and len(sites) == count
    library,_ = load_library(cfg['sidechain-library'])
    pools = candidate_pools(sites,library,cfg)
    assert math.prod(map(len,pools)) == 2**count
    for ordinal in range(2**count):
        choices = combination_at(pools,ordinal)
        product,ph = assemble(mol,h,sites,choices,f'test{ordinal}')
        assert product.GetNumHeavyAtoms() == 6+sum(s!='-H' for s in choices)
        assert ph['parent_choices'] == dict(zip(sites,choices))
        assert len({r['fragment'] for r in ph['atoms'] if r['added_iteration']==1}) == sum(s!='-H' for s in choices)
        assert len({r['uid'] for r in ph['atoms']}) == len(ph['atoms'])


def test_mixed_radix_counts_and_site_specific_candidates_are_not_m_times_n(inputs):
    _,cfg,mol,h = inputs
    sites = growth_sites(mol,h)
    library,_ = load_library(cfg['sidechain-library'])
    cfg['site-candidates'] = {sites[0]:['-H','*C'],sites[1]:['-H','*C','[*:7]F']}
    pools = candidate_pools(sites,library,cfg)
    assert len(pools[0])==2 and len(pools[1])==3
    assert [combination_at(pools,i) for i in range(6)] == [list(p) for p in itertools.product(*pools)]
    with pytest.raises(IndexError):
        combination_at(pools,6)


def test_iteration_one_space_is_not_pruned_by_diagnostic_methyl_probes(tmp_path,monkeypatch):
    _,cfg=toy(tmp_path/'all_initial',0,3)
    cfg['sidechains-per-site']=2
    db,_=stage.prepare_project(cfg);database.growth_start_session(db,cfg)
    job=database.growth_claim(db,0,2,1);database.growth_save_input(db,job['parent_pose'],{},True,{})
    job=database.growth_claim(db,1,2,1)
    monkeypatch.setattr(stage,'methyl_probe',lambda *args:dict(status=PoseVerdict.CLASH_REJECT,reason='synthetic diagnostic',pose=None))
    library,_=load_library(cfg['sidechain-library'])
    stage.plan_parent(db,job,cfg,library,None,time.time()+30)
    parent_row=database.growth_parent(db,job['parent_pose'])
    assert parent_row['plan']['total']==8 and len(parent_row['plan']['sites'])==3
    assert all(not r['gates_enumeration'] for uid,r in parent_row['probes'].items() if uid!='__baseline__')


@pytest.mark.parametrize('fragment',['[*]C','[*:42]C','*C'])
def test_attachment_spellings_normalize(fragment):
    assert normalize(fragment) == '*C'


@pytest.mark.parametrize('fragment',['-H','H','[H]','[*][H]','[*:9][H]'])
def test_hydrogen_replacement_is_a_real_candidate(fragment):
    assert normalize(fragment) == '-H'


@pytest.mark.parametrize('smiles',['*CC*','*=C','CC','not_smiles','*C.C'])
def test_invalid_fragments_are_rejected(smiles):
    with pytest.raises(ValueError):
        normalize(smiles)


def test_hydrogen_choice_keeps_atoms_and_does_not_regrow_old_frontier(inputs):
    _,_,mol,h = inputs
    sites = growth_sites(mol,h)
    product,ph = assemble(mol,h,sites,['-H']*len(sites),'keepH')
    assert provenance.canonical_smiles(product) == provenance.canonical_smiles(mol)
    assert [r['uid'] for r in ph['atoms']] == [r['uid'] for r in h['atoms']]
    assert not provenance.frontier_uids(ph)


@pytest.mark.parametrize('smiles,uid,expected_h',[('c1ccccc1N','a6',1),('Cc1ccccc1','a0',2)])
def test_only_one_h_per_heavy_atom_is_replaced(smiles,uid,expected_h):
    core = tuple(range(6)) if smiles.startswith('c1') else (1,2,3,4,5,6)
    mol,h = parent(smiles,core)
    before = {a.GetProp('_mn_uid'):a.GetTotalNumHs() for a in mol.GetAtoms()}
    product,ph = assemble(mol,h,[uid],['*C'],'oneH')
    anchor = next(a for a in product.GetAtoms() if a.GetProp('_mn_uid')==uid)
    assert anchor.GetTotalNumHs() == expected_h == before[uid]-1
    assert product.GetNumHeavyAtoms() == mol.GetNumHeavyAtoms()+1
    assert {r['uid'] for r in h['atoms']} <= {r['uid'] for r in ph['atoms']}


def test_n_o_s_and_aromatic_h_are_not_arbitrarily_excluded():
    mol,h = parent('c1ccccc1NCC(O)S',tuple(range(6)))
    sites = set(growth_sites(mol,h))
    elements = {a.GetAtomicNum() for a in mol.GetAtoms() if a.GetProp('_mn_uid') in sites}
    assert {6,7,8,16} <= elements
    mol,h = parent('Cc1cc[nH]c1')
    sites = set(growth_sites(mol,h))
    assert any(a.GetIsAromatic() and a.GetProp('_mn_uid') in sites for a in mol.GetAtoms())


def test_symmetric_growth_positions_remain_distinct_until_complete_product_deduplication():
    mol,h = parent('Cc1ccccc1')
    sites = growth_sites(mol,h)
    assert len(sites) == 5
    pools = [['-H','*C'] for _ in sites]
    products = [provenance.canonical_smiles(assemble(mol,h,sites,combination_at(pools,i),str(i))[0]) for i in range(32)]
    assert len(products) == 32 and len(set(products)) < 32


def test_every_new_fragment_is_tracked_and_only_next_iteration_becomes_frontier(tmp_path):
    _,cfg = toy(tmp_path/'two',1,2)
    mol,h = provenance.load_parents(cfg['parents'],cfg['history'])[0]
    sites = growth_sites(mol,h)
    assert len(sites) == 4
    product,ph = assemble(mol,h,sites,['*C']*4,'allfour')
    assert len(provenance.frontier_uids(ph)) == 4
    assert len({r['fragment'] for r in ph['atoms'] if r['added_iteration']==2}) == 4
    assert len(growth_sites(product,ph)) == 4
    assert not set(growth_sites(product,ph)) & set(sites)
    assert len(h['atoms']) == mol.GetNumHeavyAtoms()  # input never mutated


def test_assembly_preserves_specified_stereo_and_all_original_bonds():
    mol,h = parent('C[C@H](F)Cl')
    before = {(b.GetBeginAtom().GetProp('_mn_uid'),b.GetEndAtom().GetProp('_mn_uid')):b.GetBondType() for b in mol.GetBonds()}
    product,_ = assemble(mol,h,['a1'],['*N'],'stereo')
    assert provenance.canonical_smiles(product) == Chem.MolToSmiles(Chem.MolFromSmiles('C[C@](N)(F)Cl'),True)
    after = {(b.GetBeginAtom().GetProp('_mn_uid'),b.GetEndAtom().GetProp('_mn_uid')):b.GetBondType() for b in product.GetBonds()}
    assert all(after.get(pair)==kind for pair,kind in before.items())


def test_history_is_required_and_ambiguous_mapping_is_not_guessed(inputs):
    _,_,mol,h = inputs
    for change in ({'atoms':[]},{'core':[]},{'schema':'missing'}):
        damaged = copy.deepcopy(h) | change
        with pytest.raises(ValueError):
            provenance.annotate(Chem.Mol(mol),damaged)
    damaged = copy.deepcopy(h)
    damaged['atoms'][1]['uid'] = damaged['atoms'][0]['uid']
    with pytest.raises(ValueError,match='unique UIDs'):
        provenance.annotate(Chem.Mol(mol),damaged)


def test_probe_directions_are_multiple_but_site_uid_is_unique(inputs):
    _,_,mol,h = inputs
    terminal = growth_sites(mol,h)[-1]
    assert len(hydrogen_directions(mol,terminal)) == 3
    assert growth_sites(mol,h).count(terminal) == 1


def positioned(smiles,xyz):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms()); conf.Set3D(True)
    for i,point in enumerate(xyz):
        conf.SetAtomPosition(i,point)
    mol.AddConformer(conf)
    return mol


def test_receptor_overlap_and_spatial_index(inputs):
    _,cfg,_,_ = inputs
    ligand = positioned('C',[(0.,0.,0.)])
    assert BumpChecker(positioned('C',[(30.,0.,0.)]),cfg).check(ligand)['passed']
    result = BumpChecker(positioned('C',[(0.,0.,0.)]),cfg).check(ligand)
    assert result['receptor_clashes']==1 and not result['passed']
    assert result['receptor_max_overlap'] == pytest.approx(3.4)


@pytest.mark.parametrize('policy,expected',[('exclude',True),('include',False),('scaled',True)])
def test_one_four_policy_is_explicit(inputs,policy,expected):
    _,cfg,_,_ = inputs
    cfg.update({'one-four-policy':policy,'one-four-scale':.5})
    ligand = positioned('CCCC',[(0,0,0),(1.5,0,0),(1.5,1.5,0),(2,0,0)])
    assert BumpChecker(positioned('C',[(30,0,0)]),cfg).check(ligand)['passed'] is expected


def test_one_two_and_one_three_excluded_but_remote_internal_collision_detected(inputs):
    _,cfg,_,_ = inputs
    ligand = positioned('CCCCC',[(0,0,0),(1.5,0,0),(3,0,0),(1.5,1.5,0),(0,0,0)])
    result = BumpChecker(positioned('C',[(30,0,0)]),cfg).check(ligand)
    assert result['internal_clashes'] >= 1
    assert any({p['atom_i'],p['atom_j']}=={0,4} for p in result['internal_examples'])


def test_real_probes_pass_but_actual_fragment_combination_collides_with_itself(inputs):
    _,cfg,mol,h = inputs
    cfg['overlap-tolerance'] = 0.0  # deliberately strict model for this regression
    checker = BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
    sites = growth_sites(mol,h)
    assert all(conformations.methyl_probe(mol,h,uid,checker,cfg,37)['status']==PoseVerdict.PASS for uid in sites)
    product,ph = assemble(mol,h,sites,['*c1ccccc1']*2,'two-real-fragments')
    result = conformations.search(product,ph,checker,cfg,37)
    assert result['status'] == PoseVerdict.CLASH_REJECT
    assert result['valid_poses'] >= 1 and not result['errors']
    assert result['clashes']['receptor_clashes'] == 0
    assert any(product.GetAtomWithIdx(pair['atom_i']).GetProp('_mn_added_iteration')=='2'
               and product.GetAtomWithIdx(pair['atom_j']).GetProp('_mn_added_iteration')=='2'
               and product.GetAtomWithIdx(pair['atom_i']).GetProp('_mn_fragment') != product.GetAtomWithIdx(pair['atom_j']).GetProp('_mn_fragment')
               for pair in result['clashes']['internal_examples'])
    assert 'not a proof' in result['reason']


def test_probe_carbon_never_remains_as_a_real_fragment_linker(inputs):
    _,cfg,mol,h = inputs
    checker = BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
    uid = growth_sites(mol,h)[0]
    assert conformations.methyl_probe(mol,h,uid,checker,cfg,37)['status'] == PoseVerdict.PASS
    product,_ = assemble(mol,h,[uid],['*F'],'real')
    assert product.GetNumHeavyAtoms() == mol.GetNumHeavyAtoms()+1
    assert sum(a.GetAtomicNum()==6 for a in product.GetAtoms()) == sum(a.GetAtomicNum()==6 for a in mol.GetAtoms())
    assert mol.GetNumHeavyAtoms()==8


def test_baseline_clash_is_reported_and_core_only_rescue_can_move_parent_sidechain(inputs):
    _,cfg,_,_ = inputs
    mol,h = parent('c1ccccc1CCCC',tuple(range(6)))
    cfg.update({'conformers':4,'torsion-trials':2})
    sites = growth_sites(mol,h)
    anchor = next(a.GetIdx() for a in mol.GetAtoms() if a.GetProp('_mn_uid')==sites[-1])
    receptor = positioned('C',[mol.GetConformer().GetAtomPosition(anchor)])
    checker = BumpChecker(receptor,cfg)
    assert not checker.check(mol)['passed']
    result = conformations.methyl_probe(mol,h,sites[-1],checker,cfg,37)
    assert result['status']==PoseVerdict.PASS
    assert result['phase']=='reference_core_fixed'
    assert result['first_phase']['status'] != PoseVerdict.PASS
    assert result['core_max_displacement'] <= cfg['core-displacement']


def test_frame_probe_relaxes_old_sidechains_while_only_initial_core_is_fixed(inputs):
    _,cfg,_,_=inputs
    mol,h=parent('c1ccccc1CCCC',tuple(range(6)))
    h.update(growth_mode='frame-cycles',role='frame',
             frame_uids=[a.GetProp('_mn_uid') for a in mol.GetAtoms()],attachments=['a9'])
    cfg.update({'conformers':4,'torsion-trials':2})
    uid=growth_sites(mol,h)[-1]
    anchor=next(a.GetIdx() for a in mol.GetAtoms() if a.GetProp('_mn_uid')==uid)
    checker=BumpChecker(positioned('C',[mol.GetConformer().GetAtomPosition(anchor)]),cfg)
    assert not checker.check(mol)['passed']
    result=conformations.methyl_probe(mol,h,uid,checker,cfg,37)
    assert result['status']==PoseVerdict.PASS
    assert result['phase']=='reference_core_fixed' and 'first_phase' not in result
    assert result['constraint_scope']=='original_reference_core'
    assert result['core_max_displacement']<=cfg['core-displacement']
    original={a.GetProp('_mn_uid'):np.array(mol.GetConformer().GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms()}
    pose=result['pose']
    moved=[np.linalg.norm(np.array(pose.GetConformer().GetAtomPosition(a.GetIdx()))-original[a.GetProp('_mn_uid')])
           for a in pose.GetAtoms() if a.HasProp('_mn_uid') and a.GetProp('_mn_uid') in original
           and a.GetProp('_mn_uid') not in {r['uid'] for r in h['core']}]
    assert max(moved)>.1


def test_frame_probe_never_adds_previous_frame_atoms_to_fixed_set(inputs,monkeypatch):
    _,cfg,mol,h=inputs
    uid=growth_sites(mol,h)[0]
    h.update(growth_mode='frame-cycles',role='frame',
             frame_uids=[a.GetProp('_mn_uid') for a in mol.GetAtoms()],attachments=[uid])
    calls=[]
    def checked_search(product,birth,checker,settings,seed,**kwargs):
        fixed=conformations.fixed_coordinates(product,birth,kwargs.get('parent_fixed'))
        assert kwargs.get('parent_fixed') is None
        assert set(fixed)=={r['uid'] for r in h['core']}
        assert set(fixed)<set(h['frame_uids'])
        calls.append(fixed)
        return dict(status=PoseVerdict.PASS,pose=None)
    monkeypatch.setattr(conformations,'search',checked_search)
    result=conformations.methyl_probe(mol,h,uid,None,cfg,37)
    assert len(calls)==1 and result['phase']=='reference_core_fixed'


@pytest.mark.parametrize('failure',['embedding','parameters','optimization'])
def test_search_failures_are_unresolved_never_clash_reject(inputs,monkeypatch,failure):
    _,cfg,mol,h = inputs
    checker = BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
    def fail(*args,**kwargs):
        raise ValueError(f'{failure} failed')
    monkeypatch.setattr(conformations,'embedded' if failure=='embedding' else 'optimize',fail)
    result = conformations.search(mol,h,checker,cfg,37)
    assert result['status']==PoseVerdict.UNRESOLVED and result['valid_poses']==0
    assert any(failure in message for message in result['errors'])


@pytest.mark.parametrize('return_code',[0,1,-1])
@pytest.mark.parametrize('field',['MMFF','UFF'])
def test_optimization_iteration_limit_is_recorded_without_hiding_real_errors(inputs,monkeypatch,return_code,field):
    _,cfg,mol,h=inputs
    cfg.update({'force-field':field,'optimization-iterations':100})
    calls=[]
    class ForceField:
        def AddFixedPoint(self,index):pass
        def Initialize(self):pass
        def Minimize(self,*,maxIts):
            calls.append(maxIts)
            return return_code
        def CalcEnergy(self):return 12.5
    monkeypatch.setattr(AllChem,field+'HasAllMoleculeParams',lambda *args:True)
    monkeypatch.setattr(AllChem,field+'GetMoleculeForceField',lambda *args:ForceField())
    if field=='MMFF':
        monkeypatch.setattr(AllChem,'MMFFGetMoleculeProperties',lambda *args:object())
    if return_code==-1:
        with pytest.raises(ValueError,match='return code -1'):
            conformations.optimize(mol,conformations.fixed_coordinates(mol,h),cfg)
    else:
        kind,energy,optimization=conformations.optimize(mol,conformations.fixed_coordinates(mol,h),cfg)
        assert (kind,energy)==(field,12.5)
        assert optimization==dict(converged=return_code==0,return_code=return_code,
                                  status='converged' if return_code==0 else 'iteration_limit',max_iterations=100)
    assert calls==[100]


@pytest.mark.parametrize('bump_passes',[True,False])
def test_iteration_limited_pose_still_gets_bump_checked_and_persisted(inputs,monkeypatch,bump_passes):
    _,cfg,mol,h=inputs
    optimization=dict(status='iteration_limit',converged=False,return_code=1,max_iterations=100)
    monkeypatch.setattr(conformations,'optimize',lambda *args,**kwargs:('MMFF',0.,optimization))
    point=(1000.,1000.,1000.) if bump_passes else tuple(mol.GetConformer().GetAtomPosition(0))
    checker=BumpChecker(positioned('C',[point]),cfg)
    result=conformations.search(mol,h,checker,cfg,37,starts=[Chem.AddHs(mol,addCoords=True)])
    assert result['status']==(PoseVerdict.PASS if bump_passes else PoseVerdict.CLASH_REJECT)
    assert result['clashes']['passed']==bump_passes
    assert result['errors']==[]
    assert result['optimization_counts']['iteration_limit']==1
    assert result['optimization_counts']['converged']==0
    key='optimization' if bump_passes else 'last_optimization'
    assert result[key]==optimization
    # Stage serialization and SQLite must retain the nonconvergence evidence.
    db,_=stage.prepare_project(cfg);database.growth_start_session(db,cfg)
    database.growth_save_plan(db,1,dict(sites=[],pools=[],total=1),[])
    job=database.growth_claim(db,2,1,1)
    database.growth_save_attempt(db,job,0,dict(stage.serial_report(result),iteration=2),None)
    saved=json.loads(database.growth_rows(db,'combichem3d_attempt')[0]['report'])
    assert saved[key]==optimization


def test_iteration_limit_does_not_hide_nonfinite_forcefield_energy(inputs,monkeypatch):
    _,cfg,mol,h=inputs;cfg['force-field']='UFF'
    class ForceField:
        def AddFixedPoint(self,index):pass
        def Initialize(self):pass
        def Minimize(self,**kwargs):return 1
        def CalcEnergy(self):return float('nan')
    monkeypatch.setattr(AllChem,'UFFHasAllMoleculeParams',lambda *args:True)
    monkeypatch.setattr(AllChem,'UFFGetMoleculeForceField',lambda *args:ForceField())
    with pytest.raises(ValueError,match='Nonfinite optimized ligand energy'):
        conformations.optimize(mol,conformations.fixed_coordinates(mol,h),cfg)


def test_search_checks_core_displacement_after_position_restraints(inputs,monkeypatch):
    _,cfg,mol,h = inputs
    cfg['core-constraint']='restraint'
    checker = BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
    def moved(candidate,*a,**kw):
        p=candidate.GetConformer().GetAtomPosition(0)
        candidate.GetConformer().SetAtomPosition(0,(p.x+1,p.y,p.z))
        return 'MMFF',0.,dict(status='iteration_limit',converged=False,return_code=1,max_iterations=100)
    monkeypatch.setattr(conformations,'optimize',moved)
    result=conformations.search(mol,h,checker,cfg,37)
    assert result['status']==PoseVerdict.UNRESOLVED
    assert any('displacement' in error for error in result['errors'])


def test_ring_and_amide_bonds_are_not_generic_torsions():
    mol = Chem.AddHs(Chem.MolFromSmiles('CC(=O)NC1CCCCC1'))
    AllChem.EmbedMolecule(mol,randomSeed=37)
    for _,b,c,_ in conformations.torsions(mol,set()):
        bond = mol.GetBondBetweenAtoms(b,c)
        assert not bond.IsInRing() and not conformations.amide_bond(bond)


def test_new_substituent_can_rotate_about_a_fixed_parent_axis_without_moving_parent():
    from rdkit.Chem import rdMolTransforms
    mol,h=parent('c1ccccc1O',tuple(range(6)))
    product,ph=assemble(mol,h,['a6'],['*C'],'oxygen')
    fixed=conformations.fixed_coordinates(product,ph,mol)
    pose=conformations.embedded(product,fixed,37,{'embed-timeout':5},float('inf'))
    indices={a.GetIdx() for a in pose.GetAtoms() if a.HasProp('_mn_uid') and a.GetProp('_mn_uid') in fixed}
    quartet=next(t for t in conformations.torsions(pose,indices) if {t[1],t[2]}=={5,6})
    before=np.array(pose.GetConformer().GetPositions())
    angle=rdMolTransforms.GetDihedralDeg(pose.GetConformer(),*quartet)
    rdMolTransforms.SetDihedralDeg(pose.GetConformer(),*quartet,angle+90)
    after=np.array(pose.GetConformer().GetPositions())
    assert np.max(np.abs(before[list(indices)]-after[list(indices)]))<1e-8
    assert np.linalg.norm(before[7]-after[7])>.1


@pytest.mark.parametrize('extra',['max-iteration = 0','processes = 385','conformers = true',
                                  'max-seconds = -1','one-four-policy = "guess"'])
def test_invalid_configuration_is_rejected(inputs,extra):
    filename,_,_,_ = inputs
    text = filename.read_text().split('[filters]')[0]
    key = extra.split('=')[0].strip()
    text = '\n'.join(line for line in text.splitlines() if not line.startswith(key+' ='))
    filename.write_text(text+'\n'+extra+'\n')
    with pytest.raises(ValueError):
        combichem_3d_settings(filename)


@pytest.mark.parametrize('processes', [17, 128, 192, 384])
def test_configuration_accepts_up_to_384_workers(inputs, processes):
    filename, _, _, _ = inputs
    text = filename.read_text()
    lines = [f'processes = {processes}' if line.startswith('processes =') else line
             for line in text.splitlines()]
    filename.write_text('\n'.join(lines) + '\n')
    assert combichem_3d_settings(filename)['processes'] == processes


def test_atomic_claims_allocate_nonoverlapping_bounded_ranges(inputs):
    _,cfg,_,_ = inputs
    db,_ = stage.prepare_project(cfg)
    database.growth_start_session(db,cfg)
    input_job=database.growth_claim(db,1,2,1)
    database.growth_save_input(db,input_job['parent_pose'],{},True,{})
    plan_job=database.growth_claim(db,2,2,1)
    database.growth_save_plan(db,plan_job['parent_pose'],dict(sites=['a'],pools=[['-H']*100],total=100),[])
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs=list(pool.map(lambda owner:database.growth_claim(db,2,2,owner),range(8)))
    intervals=[set(range(int(j['cursor']),int(j['stop']))) for j in jobs]
    assert all(len(interval)==2 for interval in intervals)
    assert len(set.union(*intervals))==16


def test_big_cartesian_counts_do_not_overflow_sqlite(inputs):
    _,cfg,_,_ = inputs
    db,_=stage.prepare_project(cfg); database.growth_start_session(db,cfg)
    job=database.growth_claim(db,1,2,1); database.growth_save_input(db,job['parent_pose'],{},True,{})
    job=database.growth_claim(db,2,2,1)
    total=25**30
    database.growth_save_plan(db,job['parent_pose'],dict(sites=['x']*30,pools=[['-H']*25]*30,total=total),[])
    assert database.growth_run_info(db)['planned_combinations']==str(total)
    batch=database.growth_claim(db,2,8,1)
    assert batch['cursor']=='0' and batch['stop']=='8'


def test_cli_real_cartesian_smoke_and_exports_preserve_poses(inputs):
    filename,cfg,_,_=inputs
    completed=subprocess.run([sys.executable,'-m','molnova','combichem-3d',str(filename)],capture_output=True,text=True,timeout=40)
    assert completed.returncode==0,completed.stderr
    output=Path(cfg['out-dir']); report=json.loads((output/'summary.json').read_text())
    assert report['attempted']==9 and report['unique_count']==8
    assert report['stop_reason']=='exhausted' and not report['partial_search']
    assert report['versions']['rdkit']
    mols=list(Chem.SDMolSupplier(str(output/'passed_poses.sdf')))
    assert len(mols)==10 and all(mol and mol.GetConformer().Is3D() for mol in mols)
    assert all(mol.GetProp('_Name').startswith('CMPID_') for mol in mols)
    histories=[json.loads(line) for line in (output/'growth_history.jsonl').read_text().splitlines()]
    assert {h['iteration'] for h in histories}=={1,2}
    assert (output/'probes.jsonl').is_file() and (output/'failures.jsonl').is_file()


@pytest.mark.parametrize('mode',['target','time'])
def test_count_and_time_termination_preserve_partial_results(inputs,mode):
    _,cfg,_,_=inputs
    cfg.update({'target-count':1,'stop-mode':mode,'max-seconds':30. if mode=='target' else .00001})
    db,first=stage.prepare_project(cfg); stage.supervise(cfg,db,first)
    info=database.growth_run_info(db)
    assert info['stop_reason']==('target_count' if mode=='target' else 'time_limit')
    assert info['unique_count']==(1 if mode=='target' else 0)
    if mode=='target':
        assert info['attempted']>=1 and database.growth_rows(db,'combichem3d_pose')


def test_existing_inputs_and_scientific_settings_are_immutable_on_resume(inputs):
    _,cfg,_,_=inputs
    db,_=stage.prepare_project(cfg)
    changed=cfg|{'overlap-tolerance':.9}
    manifest=Path(cfg['out-dir'])/'input_manifest.json'
    saved_manifest=manifest.read_bytes()
    with pytest.raises(RuntimeError,match='Frozen.*overlap-tolerance') as error:
        stage.prepare_project(changed)
    assert '[combichem-3d].out-dir' in str(error.value)
    assert '--export-only' in str(error.value)
    assert manifest.read_bytes()==saved_manifest
    assert database.growth_run_info(db)['settings']['overlap-tolerance']==cfg['overlap-tolerance']


def test_implementation_fingerprint_error_identifies_changed_file(inputs,monkeypatch):
    _,cfg,_,_=inputs
    db,_=stage.prepare_project(cfg)
    manifest=Path(cfg['out-dir'])/'input_manifest.json'
    saved=manifest.read_bytes()
    original_read=Path.read_bytes
    def changed_source(path):
        content=original_read(path)
        return content+b'\n# changed calculator\n' if path.name=='cycles.py' else content
    monkeypatch.setattr(Path,'read_bytes',changed_source)
    with pytest.raises(RuntimeError,match=r'implementation_sha256.cycles.py'):
        stage.prepare_project(cfg)
    assert manifest.read_bytes()==saved
    assert database.growth_run_info(db)['attempted']==0


def test_old_project_database_cannot_be_repurposed(inputs):
    _,cfg,_,_=inputs
    Path(cfg['out-dir']).mkdir()
    with database.growth_connection(Path(cfg['out-dir'])/'combichem3d.sqlite') as conn:
        from molnova import _core
        _core.create_sqlite_schema(conn)
    with pytest.raises(RuntimeError,match='legacy project'):
        stage.prepare_project(cfg)


@pytest.mark.parametrize('prune',[False,True])
def test_output_filter_rejection_does_not_prune_intermediates_unless_requested(inputs,prune):
    _,cfg,_,_=inputs
    cfg.update({'filters':{'mw':[0.,1.]},'prune-intermediates':prune,'sidechains-per-site':2})
    db,first=stage.prepare_project(cfg); stage.supervise(cfg,db,first)
    info=database.growth_run_info(db)
    assert info['unique_count']==0
    assert info['attempted']==(0 if prune else 4)
    assert database.growth_rows(db,'combichem3d_pose')


def test_same_final_smiles_retains_different_frontier_histories(tmp_path):
    _,cfg=toy(tmp_path/'symmetric',0,2)
    cfg.update({'sidechains-per-site':2,'processes':2})
    db,first=stage.prepare_project(cfg); stage.supervise(cfg,db,first)
    info=database.growth_run_info(db)
    assert info['attempted']==4 and info['unique_count']==2
    compounds=database.growth_rows(db,'compound')
    cid=next(r['id'] for r in compounds if r['smiles']=='Cc1ccccc1')
    poses=[r for r in database.growth_rows(db,'combichem3d_pose') if r['compound_id']==cid]
    assert len(poses)==2
    histories=[json.loads(r['history']) for r in poses]
    assert histories[0]['parent_choices'] != histories[1]['parent_choices']
    assert provenance.frontier_uids(histories[0]) != provenance.frontier_uids(histories[1])


def test_resume_after_target_stop_never_repeats_committed_combinations(inputs):
    _,cfg,_,_=inputs
    cfg.update({'target-count':1,'stop-mode':'target'})
    db,first=stage.prepare_project(cfg); stage.supervise(cfg,db,first)
    before=database.growth_rows(db,'combichem3d_attempt')
    assert database.growth_run_info(db)['stop_reason']=='target_count'
    cfg.update({'target-count':10000,'stop-mode':'time'})
    stage.prepare_project(cfg); stage.supervise(cfg,db,first)
    after=database.growth_rows(db,'combichem3d_attempt')
    assert len(after)==9 and len({r['ordinal'] for r in after})==9
    assert all(row in after for row in before)
    assert database.growth_run_info(db)['unique_count']==8


def test_resume_rearms_only_unfinished_range_and_preserves_failure_evidence(inputs):
    _,cfg,_,_=inputs
    db,_=stage.prepare_project(cfg);database.growth_start_session(db,cfg)
    job=database.growth_claim(db,1,2,1);database.growth_save_input(db,job['parent_pose'],{},True,{})
    job=database.growth_claim(db,2,2,1)
    database.growth_save_plan(db,job['parent_pose'],dict(sites=['x'],pools=[['-H','*C']],total=2),[])
    batch=database.growth_claim(db,2,2,1)
    database.growth_save_attempt(db,batch,0,dict(status='UNRESOLVED',reason='synthetic interrupted geometry',iteration=2))
    database.growth_request_stop(db,'interrupted');database.growth_end_session(db,'interrupted')
    database.growth_start_session(db,cfg)
    resumed=database.growth_claim(db,2,2,2)
    assert resumed['id']==batch['id'] and resumed['cursor']=='1'
    assert len(database.growth_rows(db,'combichem3d_attempt'))==1


def test_defaults_use_one_absolute_iteration(inputs):
    filename,_,_,_=inputs
    filename.write_text('\n'.join(line for line in filename.read_text().splitlines() if not line.startswith('max-iteration ='))+'\n')
    cfg=combichem_3d_settings(filename)
    assert cfg['max-iteration']==1
    db,first=stage.prepare_project(cfg);stage.supervise(cfg,db,first)
    assert database.growth_run_info(db)['attempted']==0


def test_worker_startup_failure_is_preserved_and_never_reported_as_exhaustion(inputs,monkeypatch):
    _,cfg,_,_=inputs
    db,first=stage.prepare_project(cfg)
    def broken(*args):
        raise RuntimeError('synthetic receptor load failure')
    monkeypatch.setattr(stage,'load_receptor',broken)
    stage.supervise(cfg,db,first)
    assert database.growth_run_info(db)['stop_reason']=='worker_error'
    records=database.growth_rows(db,'combichem3d_event')
    assert records and 'synthetic receptor load failure' in records[0]['report']


def test_deadline_and_insufficient_valid_poses_are_unresolved(inputs):
    _,cfg,mol,h=inputs
    receptor=positioned('C',[mol.GetConformer().GetAtomPosition(0)])
    checker=BumpChecker(receptor,cfg)
    cfg['minimum-valid-poses']=2
    result=conformations.search(mol,h,checker,cfg,37)
    assert result['valid_poses']==1 and result['status']==PoseVerdict.UNRESOLVED
    result=conformations.search(mol,h,checker,cfg,37,deadline=time.time()-1)
    assert result['status']==PoseVerdict.UNRESOLVED and result['valid_poses']==0


def test_mixed_embedding_failure_and_clash_does_not_become_clash_reject(inputs,monkeypatch):
    _,cfg,mol,h=inputs
    cfg['conformers']=2
    checker=BumpChecker(positioned('C',[mol.GetConformer().GetAtomPosition(0)]),cfg)
    original=conformations.embedded
    calls=[]
    def second_fails(*args):
        calls.append(1)
        if len(calls)==2:
            raise ValueError('Synthetic embedding failure')
        return original(*args)
    monkeypatch.setattr(conformations,'embedded',second_fails)
    result=conformations.search(mol,h,checker,cfg,37)
    assert result['valid_poses']==1 and result['status']==PoseVerdict.UNRESOLVED


def test_ctrl_c_preserves_committed_results_and_exports_deferred_work(tmp_path):
    filename,cfg=toy(tmp_path/'interrupt',1,4)
    text=filename.read_text().replace('processes = 2','processes = 1').replace('max-seconds = 30.0','max-seconds = 30.0\ngrace-seconds = 0.5')
    filename.write_text(text)
    process=subprocess.Popen([sys.executable,'-m','molnova','combichem-3d',str(filename)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    db=Path(cfg['out-dir'])/'combichem3d.sqlite'
    ready=False
    try:
        limit=time.monotonic()+15
        while time.monotonic()<limit and process.poll() is None:
            if db.exists():
                try:
                    ready=database.growth_run_info(db)['attempted']>=1
                except Exception:
                    pass  # DB schema may not yet be committed during startup.
                if ready:
                    break
            time.sleep(.02)
        assert ready,'Artificial job did not reach a committed product in time'
        process.send_signal(signal.SIGINT)
        stdout,stderr=process.communicate(timeout=10)
        assert process.returncode==0,(stdout,stderr)
        summary=json.loads((Path(cfg['out-dir'])/'summary.json').read_text())
        assert summary['stop_reason']=='interrupted' and summary['partial_search']
        assert summary['attempted']>=1 and summary['pose_count']>=2
        assert (Path(cfg['out-dir'])/'poses.sdf').is_file()
        assert 'deferred_' in (Path(cfg['out-dir'])/'failures.jsonl').read_text()
    finally:
        if process.poll() is None:
            process.kill();process.wait()


def test_time_mode_continues_after_target_count(inputs):
    _,cfg,_,_=inputs
    cfg.update({'stop-mode':'time','target-count':1})
    db,first=stage.prepare_project(cfg);stage.supervise(cfg,db,first)
    info=database.growth_run_info(db)
    assert info['attempted']==9 and info['unique_count']==8 and info['stop_reason']=='exhausted'


def test_explicit_input_hydrogen_coordinates_are_preserved(inputs):
    _,_,mol,h=inputs
    explicit=Chem.AddHs(mol,addCoords=True)
    coordinates=np.array(explicit.GetConformer().GetPositions())
    restored,rh=provenance.annotate(explicit,h)
    assert restored.GetNumAtoms()==explicit.GetNumAtoms()
    assert np.array_equal(restored.GetConformer().GetPositions(),coordinates)
    assert len(rh['pose_coordinates'])==explicit.GetNumAtoms()


def test_accepted_pose_includes_optimized_hydrogen_coordinates(inputs):
    _,cfg,mol,h=inputs
    result=conformations.search(mol,h,BumpChecker(load_receptor(cfg['receptor'],cfg),cfg),cfg,37)
    assert result['status']==PoseVerdict.PASS
    assert result['pose'].GetNumAtoms()>result['pose'].GetNumHeavyAtoms()
    assert conformations.core_deviation(result['pose'],h)<=cfg['core-displacement']


def test_property_failure_preserves_acquired_geometry_without_treating_it_as_a_clash(inputs):
    _,cfg,_,_=inputs
    cfg['prune-intermediates']=True
    db,_=stage.prepare_project(cfg);database.growth_start_session(db,cfg)
    job=database.growth_claim(db,1,2,1);database.growth_save_input(db,job['parent_pose'],{},True,{})
    job=database.growth_claim(db,2,2,1)
    row=database.growth_parent(db,job['parent_pose']);mol=stage.restore_parent(row)
    sites=growth_sites(mol,row['history'])
    database.growth_save_plan(db,job['parent_pose'],dict(sites=sites,pools=[['*C']]*len(sites),total=1),[])
    batch=database.growth_claim(db,2,2,1)
    class Broken:
        def evaluate(self,mol):
            raise ValueError('Synthetic descriptor failure')
    stage.evaluate_batch(db,batch,cfg,BumpChecker(load_receptor(cfg['receptor'],cfg),cfg),Broken(),time.time()+30)
    attempt=database.growth_rows(db,'combichem3d_attempt')[0]
    assert attempt['status']=='UNRESOLVED' and attempt['failed_stage']=='property_evaluation'
    report=json.loads(attempt['report'])
    assert report['geometry_status']=='PASS' and not report['intermediate_pruned']
    assert len(database.growth_rows(db,'combichem3d_pose'))==2


@pytest.mark.parametrize('waters,cofactors,expected',[(False,True,2),(True,True,3),(False,False,1)])
def test_receptor_water_and_cofactor_policies(inputs,tmp_path,waters,cofactors,expected):
    _,cfg,_,_=inputs
    rw=Chem.RWMol()
    for number,residue,hetero in [(6,'ALA',False),(8,'HOH',True),(30,'ZN',True)]:
        atom=Chem.Atom(number)
        info=Chem.AtomPDBResidueInfo()
        info.SetName(f'{atom.GetSymbol():>4}');info.SetResidueName(residue);info.SetResidueNumber(rw.GetNumAtoms()+1)
        info.SetIsHeteroAtom(hetero);atom.SetMonomerInfo(info)
        rw.AddAtom(atom)
    mol=rw.GetMol();conf=Chem.Conformer(3);conf.Set3D(True)
    for i in range(3):conf.SetAtomPosition(i,(i*5.,0.,0.))
    mol.AddConformer(conf);path=tmp_path/'receptor.pdb';Chem.MolToPDBFile(mol,str(path))
    cfg.update({'include-water':waters,'include-cofactors':cofactors})
    assert load_receptor(path,cfg).GetNumAtoms()==expected


def test_linux_lock_stays_owned_by_orphaned_local_calculator(tmp_path):
    db=tmp_path/'lock.sqlite'
    code='''
import multiprocessing,time
from molnova import database
with database.stage_lock(DB,'combichem3d'):
    child=multiprocessing.get_context('fork').Process(target=time.sleep,args=(10,))
    child.start()
    print(child.pid,flush=True)
    time.sleep(20)
'''.replace('DB',repr(str(db)))
    controller=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    child=None
    try:
        child=int(controller.stdout.readline().strip())
        controller.kill();controller.wait(timeout=5)
        with pytest.raises(RuntimeError,match='already running'):
            with database.stage_lock(db,'combichem3d'):
                pass
    finally:
        if controller.poll() is None:controller.kill();controller.wait()
        if child:
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
        controller.stdout.close();controller.stderr.close()


def test_supervisor_enforces_deadline_and_terminates_slow_calculator_after_grace(inputs,monkeypatch):
    _,cfg,_,_=inputs
    cfg.update({'max-seconds':.2,'grace-seconds':.05})
    db,first=stage.prepare_project(cfg)
    original=stage.Evaluator
    marker=Path(cfg['out-dir'])/'slow-worker-entered'
    class Slow(original):
        def __init__(self,*args,**kwargs):
            marker.write_text('entered')
            time.sleep(3)
            super().__init__(*args,**kwargs)
    monkeypatch.setattr(stage,'Evaluator',Slow)
    started=time.monotonic();stage.supervise(cfg,db,first)
    assert time.monotonic()-started<2
    assert marker.exists()
    assert database.growth_run_info(db)['stop_reason']=='time_limit'


def shared_project(inputs,tmp_path,section=''):
    _,cfg,_,_=inputs
    path=tmp_path/'project.toml'
    path.write_text(f'''reference-pose = "{cfg["parents"]}"
amber-receptor = "{cfg['receptor']}"
parents = "{cfg['parents']}"
parent-history = "{cfg['history']}"
sidechain-library = "{cfg['sidechain-library']}"
out-dir = "legacy-output"
max-iteration = 100
target-count = 12345
host = "remote:128"
[combichem-3d]
growth-mode = "frontier"
sidechain-min-heavy-atoms = 1
sidechain-max-heavy-atoms = 1000
sidechain-selection = "ordered"
sidechain-include-h = true
sidechains-per-site = 3
max-iteration = 2
{section}
''')
    db=tmp_path/'legacy-output/project.sqlite'
    db.parent.mkdir()
    with database.growth_connection(db) as conn:
        from molnova import _core
        _core.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('ref','CC',0,'reference')")
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('generated','CCC',1,'generated')")
    return path,db


def test_existing_project_toml_supplies_inputs_and_isolates_3d_output(inputs,tmp_path):
    filename,source=shared_project(inputs,tmp_path)
    cfg=combichem_3d_settings(filename)
    assert cfg['receptor']==inputs[1]['receptor'] and cfg['parents']==inputs[1]['parents']
    assert cfg['sidechain-library']==inputs[1]['sidechain-library']
    assert cfg['history']==inputs[1]['history']
    assert cfg['target-count']==12345 and cfg['max-iteration']==2
    assert cfg['out-dir']==str(source.parent/'combichem-3d')
    before=source.read_bytes()
    db,_=stage.prepare_project(cfg)
    assert db!=source and source.read_bytes()==before
    assert Path(cfg['reference-smiles']).read_text()=='CC\tCMPID_1\n'
    assert database.growth_project_references(source)==[(1,'CC')]


def test_section_overrides_are_relative_to_input_toml_and_typo_checked(inputs,tmp_path):
    filename,_=shared_project(inputs,tmp_path,'receptor = "inputs/receptor.pdb"\nprocesses = 4')
    cfg=combichem_3d_settings(filename)
    assert cfg['receptor']==str(tmp_path/'inputs/receptor.pdb') and cfg['processes']==4
    filename.write_text(filename.read_text()+'conformerz = 3\n')
    with pytest.raises(ValueError,match='Unknown'):
        combichem_3d_settings(filename)


def test_shared_project_can_use_explicit_reference_smiles_without_reading_db(inputs,tmp_path):
    filename,source=shared_project(inputs,tmp_path,f'reference-smiles = "{inputs[1]["reference-smiles"]}"')
    cfg=combichem_3d_settings(filename)
    assert 'reference-source-db' not in cfg
    assert cfg['reference-smiles']==inputs[1]['reference-smiles']


def test_project_reference_cache_is_preserved_when_reference_rows_change(inputs,tmp_path):
    filename,source=shared_project(inputs,tmp_path)
    cfg=combichem_3d_settings(filename);stage.prepare_project(cfg)
    cache=Path(cfg['reference-smiles']);before=cache.read_text()
    with database.growth_connection(source) as conn:
        conn.execute("UPDATE compound SET smiles='CO' WHERE iteration=0")
    with pytest.raises(RuntimeError,match='reference identities changed'):
        stage.prepare_project(combichem_3d_settings(filename))
    assert cache.read_text()==before


def test_project_reference_mcs_can_supply_query_but_never_infers_core_mapping(inputs):
    _,cfg,mol,h=inputs
    document=json.loads(Path(cfg['history']).read_text())
    del document['parents'][0]['core_smarts']
    Path(cfg['history']).write_text(json.dumps(document))
    cfg['reference-mcs']='c1ccccc1'
    db,_=stage.prepare_project(cfg)
    assert database.growth_parent(db,1)['history']['core_smarts']=='c1ccccc1'


def test_maestro_inputs_require_explicit_coordinate_preserving_exports(inputs,tmp_path):
    filename,_=shared_project(inputs,tmp_path,'receptor = "prepared.maegz"')
    (tmp_path/'prepared.maegz').write_text('native file placeholder')
    with pytest.raises(ValueError,match='receptor PDB/SDF'):
        stage.prepare_project(combichem_3d_settings(filename))


def test_project_default_parent_pose_path_and_flat_controls_are_reused(inputs,tmp_path):
    filename,source=shared_project(inputs,tmp_path)
    text=filename.read_text()
    text='\n'.join(line for line in text.splitlines() if not line.startswith('parents ='))
    text=text.replace('[combichem-3d]','conformers = 5\n[combichem-3d]')
    filename.write_text(text+'\n')
    parent_path=source.parent/'iter01/06-glide/best_poses.sdf'
    parent_path.parent.mkdir(parents=True)
    parent_path.write_bytes(Path(inputs[1]['parents']).read_bytes())
    cfg=combichem_3d_settings(filename)
    assert cfg['parents']==inputs[1]['parents'] and cfg['conformers']==5


def test_project_can_extract_references_from_sdf_without_creating_source_db(inputs,tmp_path):
    filename,source=shared_project(inputs,tmp_path)
    source.unlink()
    filename.write_text(filename.read_text().replace('reference-pose = "references.maegz"',
                       f'reference-pose = "{inputs[1]["parents"]}"'))
    cfg=combichem_3d_settings(filename)
    stage.prepare_project(cfg)
    assert 'CCc1ccccc1' in Path(cfg['reference-smiles']).read_text()
    assert not source.exists()


def test_existing_input_toml_cli_reuses_shared_data_end_to_end(inputs,tmp_path):
    filename,source=shared_project(inputs,tmp_path,'processes = 2\nmax-seconds = 30.0\nstop-mode = "time"')
    before=source.read_bytes()
    result=subprocess.run([sys.executable,'-m','molnova','combichem-3d',str(filename)],
                          capture_output=True,text=True,timeout=40)
    assert result.returncode==0,(result.stdout,result.stderr)
    report=json.loads((source.parent/'combichem-3d/summary.json').read_text())
    assert report['attempted']==9 and report['unique_count']==8
    assert source.read_bytes()==before


def test_project_native_sources_prefer_reference_pose_and_grid(inputs,tmp_path):
    filename,_=shared_project(inputs,tmp_path)
    filename.write_text('dock-grid="input/grid.zip"\n'+filename.read_text().replace(
        f'reference-pose = "{inputs[1]["parents"]}"','reference-pose="input/poses.maegz"'))
    cfg=combichem_3d_settings(filename)
    assert cfg['parents']==str(tmp_path/'input/poses.maegz')
    assert cfg['receptor']==str(tmp_path/'input/grid.zip')


@pytest.mark.parametrize('names',[[],['a_recep.mae','b_recep.mae']])
def test_native_grid_rejects_ambiguous_or_absent_receptor(tmp_path,names):
    import zipfile
    from molnova.chemistry.combichem3d.native_extract import receptor_member
    path=tmp_path/'grid.zip'
    with zipfile.ZipFile(path,'w') as archive:
        for name in names: archive.writestr(name,'test')
    with pytest.raises(ValueError,match='exactly one'):
        receptor_member(path)


def test_native_cache_validates_coordinates_and_retains_changed_sources(inputs,tmp_path,monkeypatch):
    from molnova.chemistry.combichem3d import inputs as native
    _,base,mol,_=inputs
    source=tmp_path/'reference.maegz';source.write_text('native placeholder')
    suite=tmp_path/'suite';suite.mkdir();(suite/'run').touch()
    cfg=dict(base,parents=str(source),schrodinger=str(suite))
    calls=[]
    def export(command,**kwargs):
        calls.append(command)
        folder=Path(command[command.index('--output')+1])
        with Chem.SDWriter(str(folder/'parents.sdf')) as writer:writer.write(mol)
        record=dict(atomic_numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
                    formal_charges=[a.GetFormalCharge() for a in mol.GetAtoms()],
                    xyz=mol.GetConformer().GetPositions().tolist(),bond_count=mol.GetNumBonds())
        (folder/'atom_source_map.json').write_text(json.dumps(dict(parents=[record])))
        return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr(native.subprocess,'run',export)
    native.materialize_native_inputs(cfg)
    assert len(calls)==1 and Path(cfg['parents']).suffix=='.sdf'
    fresh=dict(base,parents=str(source),schrodinger=str(suite))
    native.materialize_native_inputs(fresh)
    assert len(calls)==1
    folder=Path(cfg['parents']).parent
    audit=json.loads((folder/'atom_source_map.json').read_text())
    audit['parents'][0]['xyz'][0][0]+=1
    with pytest.raises(ValueError,match='reordered/moved'):
        native.validate_exports(folder,audit)
    source.write_text('changed source')
    with pytest.raises(RuntimeError,match='sources changed'):
        native.materialize_native_inputs(dict(base,parents=str(source),schrodinger=str(suite)))
    assert Path(cfg['parents']).exists()
    source.write_text('native placeholder')
    Path(cfg['parents']).write_text('tampered')
    with pytest.raises(RuntimeError,match='cache changed'):
        native.materialize_native_inputs(dict(base,parents=str(source),schrodinger=str(suite)))


def test_native_reference_fallback_uses_exported_parent_sdf(inputs):
    _,cfg,_,_=inputs
    cfg['reference-smiles']=str(Path(cfg['out-dir'])/'input-cache/references.smi')
    cfg['native-parent-source']='/input/reference.maegz'
    cfg['reference-source-poses']=cfg['native-parent-source']
    stage.materialize_project_references(cfg)
    assert 'CCc1ccccc1' in Path(cfg['reference-smiles']).read_text()


def test_project_missing_history_uses_inferred_cache_and_explicit_wins(inputs,tmp_path):
    filename,_=shared_project(inputs,tmp_path)
    filename.write_text('\n'.join(l for l in filename.read_text().splitlines() if not l.startswith('parent-history ='))+'\nreference-mcs="c1ccccc1"\n')
    cfg=combichem_3d_settings(filename)
    assert cfg['history-source']=='inferred'
    assert cfg['inferred-parent-iteration']==1
    db,_=stage.prepare_project(cfg)
    history=database.growth_parent(db,1)['history']
    assert history['provenance']['status']=='INFERRED'
    assert growth_sites(provenance.restore(Chem.MolFromMolBlock(database.growth_parent(db,1)['molblock'],removeHs=False),history),history)
    filename.write_text(filename.read_text()+f'history="{inputs[1]["history"]}"\n')
    cfg=combichem_3d_settings(filename)
    assert 'history-source' not in cfg and cfg['history']==inputs[1]['history']


@pytest.mark.parametrize('iteration',[1,3])
def test_inference_groups_components_preserves_pose_and_tracks_all_fragments(iteration):
    from molnova.chemistry.combichem3d.inference import infer_parent
    mol=Chem.AddHs(Chem.MolFromSmiles('CCc1ccc(OCC)cc1'))
    AllChem.EmbedMolecule(mol,randomSeed=42)
    before=mol.GetConformer().GetPositions().copy()
    h=infer_parent(mol,'c1cc([*:1])ccc1[*:0]',iteration=iteration)
    assert h['iteration']==iteration and len(h['provenance']['fragments'])==2
    assert len(h['attachments'])==2
    assert np.array_equal(before,mol.GetConformer().GetPositions())
    annotated,h=provenance.annotate(mol,h)
    assert len({r['fragment'] for r in h['atoms'] if r['added_iteration']==iteration})==2
    assert all(r['added_iteration']==(0 if r['branch']=='core' else iteration) for r in h['atoms'])
    assert len(growth_sites(annotated,h))==4
    assert h['provenance']['ambiguous_core_mapping']
    assert infer_parent(mol,'c1cc([*:1])ccc1[*:0]',iteration=iteration)==h


def test_inference_no_match_does_not_fabricate_core(inputs):
    from molnova.chemistry.combichem3d.inference import infer_parent
    with pytest.raises(ValueError,match='does not match'):
        infer_parent(inputs[2],'[Si]')


def test_inferred_cache_is_preserved_if_assumption_changes(inputs):
    from molnova.chemistry.combichem3d.inference import materialize_inferred_history
    _,cfg,_,_=inputs
    cfg=dict(cfg,**{'history-source':'inferred','reference-mcs':'c1ccccc1',
                    'history':str(Path(cfg['out-dir'])/'input-cache/inferred-history.json')})
    materialize_inferred_history(cfg)
    previous=Path(cfg['history']).read_bytes()
    cfg['inferred-parent-iteration']=2
    with pytest.raises(RuntimeError,match='Inferred history changed'):
        materialize_inferred_history(cfg)
    assert Path(cfg['history']).read_bytes()==previous


def test_project_can_disable_automatic_history_inference(inputs,tmp_path):
    filename,_=shared_project(inputs,tmp_path,'infer-history=false')
    filename.write_text('\n'.join(l for l in filename.read_text().splitlines() if not l.startswith('parent-history =')))
    with pytest.raises(ValueError,match='history must'):
        combichem_3d_settings(filename)


@pytest.mark.parametrize('debug',[False,True,'trace'])
def test_cli_logs_progress_worker_diagnostics_and_appends_on_resume(inputs,debug):
    filename,cfg,_,_=inputs
    filename.write_text('log-interval=0.01\n'+filename.read_text())
    # Match the installed console script's sys.exit(main()) contract as well
    # as forwarding the debug flag into independent child workers.
    command=[sys.executable,'-c','import sys; from molnova.cli import main; sys.exit(main(sys.argv[1:]))',
             'combichem-3d',str(filename)]
    flags=['--trace'] if debug=='trace' else (['--debug'] if debug else [])
    result=subprocess.run(command+flags,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    for text in ('starting config=','preflight:','sidechain library ready:',
                 'FRAME CMPID_', 'RESULT CMPID_', 'iteration 2 result:', 'progress iteration=',
                 'CLASH_REJECT=','UNRESOLVED=','remaining=','exports ready'):
        assert text in result.stdout
    assert ('probe start parent_pose=' in result.stdout)==(debug=='trace')
    assert ('product start parent_pose=' in result.stdout)==(debug=='trace')
    assert ('baseline parent_pose=' in result.stdout)==bool(debug)
    if debug!='trace':
        assert 'errors=[' not in result.stdout and 'internal_examples' not in result.stdout
    log=Path(cfg['out-dir'])/'run.log'
    original=log.read_text()
    assert 'pid=' in original and 'progress iteration=' in original
    assert original.count('starting config=')==1
    inventory=[json.loads(line) for line in (Path(cfg['out-dir'])/'iteration02-structures.jsonl').read_text().splitlines()]
    assert inventory[0]['kind']=='summary' and inventory[0]['frames']==1
    assert inventory[0]['attempted']==9 and not inventory[0]['partial']
    frames=[r for r in inventory if r['kind']=='frame']
    products=[r for r in inventory if r['kind']=='result']
    assert len(frames)==1 and frames[0]['iteration']==1 and frames[0]['core_smarts']
    assert products and all(r['iteration']==2 and r['parent_pose']==1 for r in products)
    assert all(Chem.MolFromSmiles(r['smiles']) is not None for r in products)
    assert len({r['compound_id'] for r in products if r['output_passed']})==inventory[0]['passed_unique']
    assert result.stdout.count('RESULT CMPID_')==min(3,inventory[0]['passed_unique'])
    summary=json.loads((Path(cfg['out-dir'])/'summary.json').read_text())
    # Changing verbosity is operational: it must not invalidate frozen chemistry
    # or reattempt completed combinations. A new invocation appends its log.
    resumed=subprocess.run(command+([] if debug else ['--debug']),capture_output=True,text=True,timeout=30)
    assert resumed.returncode==0,resumed.stderr
    assert 'resume validated' in resumed.stdout
    assert log.read_text().startswith(original)
    assert log.read_text().count('starting config=')==2
    assert resumed.stdout.rstrip().endswith('===== 실행 결과 분석 END =====')
    assert log.read_text().rstrip().endswith('===== 실행 결과 분석 END =====')
    assert json.loads((Path(cfg['out-dir'])/'summary.json').read_text())['attempted']==summary['attempted']


@pytest.mark.parametrize('setting',['log-interval=0','log-interval=-1','log-level="VERBOSE"','log-level=true'])
def test_logging_settings_are_validated(inputs,setting):
    filename,_,_,_=inputs
    filename.write_text(setting+'\n'+filename.read_text())
    with pytest.raises(ValueError,match='log-'):
        combichem_3d_settings(filename)


def test_sidechain_size_filter_counts_heavy_atoms_without_dummy_or_hydrogens(tmp_path):
    filename=tmp_path/'sizes.smi'
    filename.write_text('*CCCC\n[*:8]CCCCC\n*CCCCCCCCCC\n*CCCCCCCCCCC\n[2H]C([*])CCCC\n-H\n')
    cfg={'sidechain-min-heavy-atoms':5,'sidechain-max-heavy-atoms':10,'sidechain-include-h':False}
    library,rejected=load_library(filename,cfg)
    assert '-H' not in library
    assert sorted(Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in library)==[5,5,10]
    assert sorted(r['heavy_atoms'] for r in rejected)==[4,11]


def test_random_sidechain_pools_are_unique_seeded_per_site_and_preserve_full_product(tmp_path):
    filename=tmp_path/'library.smi'
    filename.write_text('\n'.join('*'+('C'*n)+x for n in range(4,9) for x in ('','O','N','F')))
    cfg={'sidechain-min-heavy-atoms':5,'sidechain-max-heavy-atoms':10,'sidechain-include-h':False,
         'sidechains-per-site':10,'sidechain-selection':'random','seed':42,'site-candidates':{}}
    library,_=load_library(filename,cfg)
    pools=candidate_pools(['a','b'],library,cfg)
    assert all(len(p)==len(set(p))==10 and '-H' not in p for p in pools)
    assert pools==candidate_pools(['a','b'],list(reversed(library)),cfg)
    assert pools!=candidate_pools(['a','b'],library,cfg|{'seed':43})
    assert pools[0]!=pools[1]
    assert len([combination_at(pools,i) for i in range(100)])==100
    assert all(5<=Chem.MolFromSmiles(s).GetNumHeavyAtoms()<=10 for p in pools for s in p)
    cfg['sidechains-per-site']=3
    assert all(len(p)==3 for p in candidate_pools(['a','b'],library,cfg))
    cfg['site-candidates']={'a':library[:4]}
    with pytest.raises(ValueError,match='exceeds sidechains-per-site'):
        candidate_pools(['a'],library,cfg)


def test_optional_h_choice_counts_toward_k(tmp_path):
    filename=tmp_path/'library.smi';filename.write_text('*CCCCC\n*CCCCCO\n*CCCCCN\n')
    cfg={'sidechain-min-heavy-atoms':5,'sidechain-max-heavy-atoms':10,'sidechain-include-h':True,
         'sidechains-per-site':3,'sidechain-selection':'random','seed':42,'site-candidates':{}}
    library,_=load_library(filename,cfg)
    pools=candidate_pools(['a'],library,cfg)
    assert len(pools[0])==3 and pools[0][0]=='-H'


def test_target_limit_is_atomic_and_preserves_finished_geometry(inputs):
    _,cfg,mol,h=inputs
    cfg.update({'target-count':2,'stop-mode':'target'})
    db,_=stage.prepare_project(cfg);database.growth_start_session(db,cfg)
    sites=growth_sites(mol,h)
    database.growth_save_plan(db,1,dict(sites=sites,pools=[['-H']]*len(sites),total=8),[])
    job=database.growth_claim(db,2,8,100)
    def finish(number):
        product,birth=assemble(mol,h,[sites[0]],['*'+('C'*(number+1))],f'cap{number}')
        payload=dict(smiles=provenance.canonical_smiles(product),molblock=Chem.MolToMolBlock(product),
                     history=birth,properties={},output_passed=True,pruned=False)
        database.growth_save_attempt(db,job,number,dict(status='PASS',reason='synthetic valid geometry',iteration=2,
                                                       output_passed=True,filter_rejections=[]),payload)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(finish,range(8)))
    info=database.growth_run_info(db)
    assert info['unique_count']==2 and info['attempted']==8 and info['stop_reason']=='target_count'
    poses=[r for r in database.growth_rows(db,'combichem3d_pose') if r['generated']]
    assert len(poses)==8 and sum(r['output_passed'] for r in poses)==2
    reports=[json.loads(r['report']) for r in database.growth_rows(db,'combichem3d_attempt')]
    assert sum(r.get('target_count_limited',False) for r in reports)==6


@pytest.mark.parametrize('extra', [
    'execution-host = "-bad"\nremote-env-script = "/env/amber.sh"',
    'execution-host = "t41"',
    'execution-host = "t41"\nremote-env-script = "relative/amber.sh"',
    'remote-env-script = "/env/amber.sh"',
])
def test_remote_execution_settings_reject_invalid_configuration(inputs, extra):
    filename, _, _, _ = inputs
    filename.write_text(filename.read_text().split('[filters]')[0] + '\n' + extra + '\n')
    with pytest.raises(ValueError):
        combichem_3d_settings(filename)


def test_locked_result_transaction_rolls_back_and_retries_without_duplicates(inputs, monkeypatch):
    import sqlite3
    _, cfg, _, _ = inputs
    db, _ = stage.prepare_project(cfg)
    database.growth_start_session(db, cfg)
    job = database.growth_claim(db, 1, 1, 1)
    database.growth_save_input(db, job['parent_pose'], {}, True, {})
    job = database.growth_claim(db, 2, 1, 1)
    database.growth_save_plan(db, job['parent_pose'], dict(sites=['a'], pools=[['-H']], total=1), [])
    job = database.growth_claim(db, 2, 1, 1)
    calls = []
    class BusyOnce(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if sql.startswith('UPDATE combichem3d_run SET attempted='):
                calls.append(sql)
                if len(calls) == 1:
                    exc = sqlite3.OperationalError('database is locked')
                    exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
                    raise exc
            return super().execute(sql, parameters)
    monkeypatch.setattr(database, 'connect', lambda path: sqlite3.connect(path, factory=BusyOnce))
    database.growth_save_attempt(db, job, 0, dict(status='CLASH_REJECT', reason='synthetic clash', iteration=2))
    assert len(calls) == 2
    assert len(database.growth_rows(db, 'combichem3d_attempt')) == 1
    assert database.growth_run_info(db)['attempted'] == 1
    assert database.growth_rows(db, 'combichem3d_batch')[0]['cursor'] == '1'


def test_idle_claim_does_not_compete_with_result_writer(inputs):
    _, cfg, _, _ = inputs
    db, _ = stage.prepare_project(cfg)
    database.growth_start_session(db, cfg)
    job = database.growth_claim(db, 1, 1, 1)
    database.growth_save_input(db, job['parent_pose'], {}, True, {})
    job = database.growth_claim(db, 2, 1, 1)
    database.growth_save_plan(db, job['parent_pose'], dict(sites=['a'], pools=[['-H']], total=1), [])
    assert database.growth_claim(db, 2, 1, 1)['kind'] == 'batch'
    with database.growth_connection(db) as writer:
        writer.execute('BEGIN IMMEDIATE')
        assert database.growth_claim(db, 2, 1, 2) == dict(kind='wait')


def test_non_lock_database_errors_are_not_retried():
    import sqlite3
    calls = []
    @database._growth_retry_locked
    def invalid():
        calls.append(1)
        raise sqlite3.OperationalError('no such table')
    with pytest.raises(sqlite3.OperationalError, match='no such table'):
        invalid()
    assert len(calls) == 1


def test_two_host_configuration_uses_192_slots_each_and_resolves_known_hosts(inputs):
    filename, _, _, _ = inputs
    extra = '''execution-host = "cpu1"
remote-env-script = "/env/amber.sh"
worker-hosts = { cpu1 = 192, cpu2 = 192 }
worker-ssh-known-hosts = "keys/known_hosts"
'''
    filename.write_text(filename.read_text().split('[filters]')[0] + '\n' + extra)
    cfg = combichem_3d_settings(filename)
    assert cfg['processes'] == 384
    assert cfg['worker-hosts'] == {'cpu1':192, 'cpu2':192}
    assert cfg['worker-ssh-known-hosts'] == str(filename.parent/'keys/known_hosts')


@pytest.mark.parametrize('hosts', ['{ cpu1 = 193 }', '{ cpu1 = true }',
    '{ cpu1 = 1, cpu2 = 1, cpu3 = 1 }', '{ cpu2 = 192 }', '{ "-bad" = 1 }'])
def test_invalid_distributed_host_capacity_is_rejected(inputs, hosts):
    filename, _, _, _ = inputs
    filename.write_text(filename.read_text().split('[filters]')[0] + '\n' +
        'execution-host = "cpu1"\nremote-env-script = "/env/amber.sh"\nworker-hosts = '+hosts+'\n')
    with pytest.raises(ValueError):
        combichem_3d_settings(filename)
