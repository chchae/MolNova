"""Frame-only growth and elite-MCS update regressions; synthetic receptor geometry."""
import copy
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import time
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from molnova import database
from molnova.config import combichem_3d_settings
from molnova.stages import combichem_3d as stage
from molnova.chemistry.combichem3d import cycles, frames, history as provenance
from molnova.chemistry.combichem3d.assembly import assemble, growth_sites
from molnova.chemistry.combichem3d.library import load_library,candidate_pools,combination_at
from molnova.chemistry.combichem3d.clashes import BumpChecker,load_receptor
from molnova.chemistry.combichem3d.output import export_results
from molnova.chemistry.combichem3d.attachment_policy import root_for_site, balanced_sites, growth_available, initial_attachment_spec
from molnova.chemistry.combichem3d.analysis import attachment_display, final_analysis_lines


@pytest.fixture
def project(tmp_path):
    path=Path(__file__).parents[1]/'examples/combichem-3d/make_toy_inputs.py'
    spec=importlib.util.spec_from_file_location('frame_toy',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    filename=module.create(tmp_path/'toy',1,2)
    cfg=combichem_3d_settings(filename)
    cfg.update({'growth-mode':'frame-cycles','reference-mcs':'c1([*:0])c([*:1])cccc1','stop-mode':'cycles',
                'sidechain-include-h':False,'sidechains-per-site':2,'max-iteration':2,
                'conformers':1,'torsion-trials':0,'optimization-iterations':1000,'processes':2})
    return cfg


def product_row(frame,history,number=0,choices=None):
    choices=choices or ['*CCCCC','*CCCCCC']
    mol,birth=assemble(frame,history,history['attachments'],choices,f'test{number}')
    AllChem.EmbedMolecule(mol,randomSeed=42+number)
    return dict(id=number+2,compound_id=number+2,smiles=provenance.canonical_smiles(mol),
                molblock=Chem.MolToMolBlock(mol),history=provenance.snapshot(mol,birth),
                properties={'qed':.9-number*.001,'similarity':.5},output_passed=True,pruned=False)


def seed_completed_cycle(cfg):
    db,first=stage.prepare_project(cfg);database.growth_start_session(db,cfg)
    row=database.growth_parent(db,1);frame=stage.restore_parent(row);history=row['history']
    database.growth_save_input(db,1,{},False,{})
    database.growth_save_plan(db,1,dict(sites=history['attachments'],pools=[['*CCCCC']]*2,total=1),[])
    job=database.growth_claim(db,1,1,1)
    payload=product_row(frame,history)
    database.growth_save_attempt(db,job,0,dict(status='PASS',reason='synthetic pose',iteration=1,output_passed=True),payload)
    database.growth_finish_batch(db,job)
    return db,frame,history,payload


def test_initial_frame_excludes_reference_sidechains_and_enumerates_100(project):
    cfg=project
    cfg.update({'max-iteration':1,'sidechains-per-site':10,'sidechain-selection':'random',
                'sidechain-min-heavy-atoms':5,'sidechain-max-heavy-atoms':10})
    Path(cfg['sidechain-library']).write_text('\n'.join('*'+'C'*n+x for n in range(5,10) for x in ('','O','N')))
    db,first=stage.prepare_project(cfg)
    assert first==0
    rows=database.growth_rows(db,'compound')
    assert len(rows)==1 and rows[0]['smiles']=='c1ccccc1'
    row=database.growth_parent(db,1);frame=stage.restore_parent(row);h=row['history']
    assert frame.GetNumHeavyAtoms()==6 and len(h['attachments'])==2
    assert cfg['reference-poses']!=cfg['parents']
    library,_=load_library(cfg['sidechain-library'],cfg)
    pools=candidate_pools(h['attachments'],library,cfg)
    for i in range(100):
        product,birth=assemble(frame,h,h['attachments'],combination_at(pools,i),str(i))
        assert product.GetNumHeavyAtoms()>=16 and birth['iteration']==1
    with pytest.raises(IndexError):combination_at(pools,100)
    stage.prepare_project(cfg) # same-object resume must still read original references


def test_constraint_reference_selection_preserves_core_map_and_uses_best_similarity(project):
    cfg=project;db,_=stage.prepare_project(cfg)
    row=database.growth_parent(db,1);frame=stage.restore_parent(row)
    ref=frames.ReferencePoses(cfg['reference-poses'],cfg['reference-mcs'])
    product,birth=assemble(frame,row['history'],row['history']['attachments'],['*CC','*CC'],'reference')
    constrained=ref.constrain(product,birth)
    assert constrained['constraint_reference']['similarity']==1.
    assert [r['uid'] for r in constrained['core']]==[r['uid'] for r in birth['core']]


def test_same_mcs_halves_20_10_5_2_then_uses_whole_single_elite(project,monkeypatch):
    cfg=project;db,_=stage.prepare_project(cfg)
    previous=database.growth_parent(db,1);frame=stage.restore_parent(previous)
    elites=[product_row(frame,previous['history'],i) for i in range(20)]
    calls=[]
    def find(mols,params):
        calls.append(len(mols));return SimpleNamespace(canceled=False,smartsString='[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1')
    monkeypatch.setattr(cycles.rdFMCS,'FindMCS',find)
    (new,h),report=cycles.select_mcs(elites,previous,cfg,time.time()+30)
    assert calls==[20,10,5,2]
    assert [a['elite_count'] for a in report['attempts']]==[20,10,5,2,1]
    # Equal molecular weights preserve QED/ID order at every reduction.
    assert all(a['elite_pose_ids']==[r['id'] for r in elites[:a['elite_count']]] for a in report['attempts'])
    assert all(a['status']=='same_as_previous_frame' for a in report['attempts'][:-1])
    assert report['attempts'][-1]['status']=='single_elite_full_structure'
    assert provenance.canonical_smiles(new)==elites[0]['smiles']
    assert new.GetNumHeavyAtoms()>frame.GetNumHeavyAtoms()


def test_mcs_timeout_discards_partial_query_then_reduces(project,monkeypatch):
    cfg=project;db,_=stage.prepare_project(cfg);previous=database.growth_parent(db,1);frame=stage.restore_parent(previous)
    elites=[product_row(frame,previous['history'],i) for i in range(2)]
    monkeypatch.setattr(cycles.rdFMCS,'FindMCS',lambda *args:SimpleNamespace(canceled=True,smartsString='invalid partial'))
    selected,report=cycles.select_mcs(elites,previous,cfg,time.time()+30)
    assert selected and report['attempts'][0]['status']=='mcs_timeout'
    assert report['attempts'][0]['mcs_smarts'] is None


@pytest.mark.parametrize('timeout',[False,True])
def test_elite_reduction_discards_lowest_molecular_weights_first(project,monkeypatch,timeout):
    cfg=project;db,_=stage.prepare_project(cfg)
    previous=database.growth_parent(db,1);frame=stage.restore_parent(previous)
    # QED order favors the smaller products; Br is heaviest despite fewer atoms.
    elites=[product_row(frame,previous['history'],i,choices=[fragment,'*C'])
            for i,fragment in enumerate(['*C','*CC','*CCC','*CCCC','*Br'])]
    original_ids=[r['id'] for r in elites]
    monkeypatch.setattr(cycles.rdFMCS,'FindMCS',lambda *args:SimpleNamespace(
        canceled=timeout,smartsString='invalid partial' if timeout else '[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'))
    (new,_),report=cycles.select_mcs(elites,previous,cfg,time.time()+30)
    attempts=report['attempts']
    assert [a['elite_pose_ids'] for a in attempts]==[original_ids,[elites[4]['id'],elites[3]['id']],[elites[4]['id']]]
    assert all(a['reduction_by']=='molecular_weight' for a in attempts)
    assert attempts[1]['molecular_weights'][0]>attempts[1]['molecular_weights'][1]
    assert provenance.canonical_smiles(new)==elites[4]['smiles']
    assert [r['id'] for r in elites]==original_ids


def test_elite_reduction_stays_within_initial_ranked_selection(project,monkeypatch):
    cfg=project;cfg['elite-count']=2
    db,_=stage.prepare_project(cfg);previous=database.growth_parent(db,1);frame=stage.restore_parent(previous)
    elites=[product_row(frame,previous['history'],i,choices=[fragment,'*C'])
            for i,fragment in enumerate(['*C','*CC','*Br'])]
    monkeypatch.setattr(cycles.rdFMCS,'FindMCS',lambda *args:SimpleNamespace(canceled=True,smartsString=''))
    _,report=cycles.select_mcs(elites,previous,cfg,time.time()+30)
    assert report['elite_pose_ids']==[elites[1]['id']]
    assert report['attempts'][0]['elite_pose_ids']==[elites[0]['id'],elites[1]['id']]


def test_changed_mcs_is_accepted_without_halving(project,monkeypatch):
    cfg=project;db,_=stage.prepare_project(cfg);previous=database.growth_parent(db,1);frame=stage.restore_parent(previous)
    elites=[product_row(frame,previous['history'],i) for i in range(2)]
    template=cycles.restored(elites[0]);query=Chem.MolToSmarts(Chem.RemoveHs(template))
    monkeypatch.setattr(cycles.rdFMCS,'FindMCS',lambda *args:SimpleNamespace(canceled=False,smartsString=query))
    selected,report=cycles.select_mcs(elites,previous,cfg,time.time()+30)
    assert selected and len(report['attempts'])==1 and report['selected_elite_count']==2


@pytest.fixture
def shared_ring_elites(project):
    """Reported CMPID_8/13 graphs, with synthetic coordinates and core UIDs."""
    db,_=stage.prepare_project(project);previous=database.growth_parent(db,1)
    smiles=[
        'c1(C2=C[C@H]3C=NN(C)[C@H]3C=C2)ccc(-c2cc3c(NC4=CC[C@H](C(=O)NC)CC4)ncnc3[nH]2)cc1',
        'c1(OCCN(C)C)ccc(-c2cc3c(NC4=CC[C@H](C(=O)NC)CC4)ncnc3[nH]2)cc1',
    ]
    rows=[]
    for number,text in enumerate(smiles):
        mol=Chem.MolFromSmiles(text)
        conf=Chem.Conformer(mol.GetNumAtoms());conf.Set3D(True)
        for i in range(mol.GetNumAtoms()):conf.SetAtomPosition(i,(float(i),float(i%3),0.))
        mol.AddConformer(conf)
        h=copy.deepcopy(previous['history']);h['iteration']=1
        core=mol.GetSubstructMatch(Chem.MolFromSmarts(h['core_smarts']))
        by_index={core[r['core_index']]:r['uid'] for r in h['core']}
        h['atoms']=[dict(index=i,uid=by_index.get(i,f'elite{number}:{i}'),
                         branch='core' if i in by_index else 'sidechain',
                         fragment='core' if i in by_index else 'sidechain',added_iteration=0 if i in by_index else 1)
                    for i in range(mol.GetNumAtoms())]
        mol,h=provenance.annotate(mol,h)
        rows.append(dict(id=number+8,compound_id=number+8,molblock=Chem.MolToMolBlock(mol),history=h))
    return project,previous,rows


def test_reported_common_six_membered_ring_is_preserved(shared_ring_elites):
    cfg,previous,elites=shared_ring_elites
    original=[copy.deepcopy(r) for r in elites]
    expected='CNC(=O)[C@H]1CC=C(Nc2ncnc3[nH]c(-c4ccccc4)cc23)CC1'
    for subset in (elites,elites[::-1]):
        (frame,h),report=cycles.select_mcs(subset,previous,cfg,time.time()+30)
        assert report['selected_elite_count']==2
        assert frame.GetNumHeavyAtoms()==26
        assert provenance.canonical_smiles(frame)==provenance.canonical_smiles(Chem.MolFromSmiles(expected))
        assert all(cycles.restored(r).HasSubstructMatch(frame,useChirality=True) for r in elites)
        template=cycles.restored(subset[0])
        assert not cycles.cuts_ring(template,report['attempts'][0]['selected_template_indices'])
        original_xyz={a.GetProp('_mn_uid'):tuple(template.GetConformer().GetAtomPosition(a.GetIdx()))
                      for a in template.GetAtoms() if a.GetAtomicNum()>1}
        assert all(tuple(frame.GetConformer().GetAtomPosition(a.GetIdx()))==original_xyz[a.GetProp('_mn_uid')]
                   for a in frame.GetAtoms())
    assert elites==original


def test_mcs_normalization_does_not_relax_opposite_stereochemistry(shared_ring_elites):
    cfg,previous,elites=shared_ring_elites
    opposite=Chem.MolFromMolBlock(elites[1]['molblock'],removeHs=False)
    centers=[a for a in opposite.GetAtoms() if a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED]
    assert len(centers)==1
    centers[0].InvertChirality()
    elites[1]['molblock']=Chem.MolToMolBlock(opposite)
    assert provenance.canonical_smiles(cycles.mcs_input(opposite))==provenance.canonical_smiles(opposite)
    (frame,_),report=cycles.select_mcs(elites,previous,cfg,time.time()+30)
    assert report['selected_elite_count']==2
    assert frame.GetNumHeavyAtoms()<26
    assert all(cycles.restored(r).HasSubstructMatch(frame,useChirality=True) for r in elites)


def test_partial_ring_query_is_rejected_even_if_fmcs_returns_it(shared_ring_elites,monkeypatch):
    cfg,previous,elites=shared_ring_elites
    # A query containing the retained phenyl core plus a dangling ring carbon.
    query='[#6]1:[#6]:[#6](-[#6]2:[#6]:[#6]3:[#6](-[#7]-[#6&R]):[#7]:[#6]:[#7]:[#6]:3:[#7]:2):[#6]:[#6]:[#6]:1'
    monkeypatch.setattr(cycles.rdFMCS,'FindMCS',lambda *args:SimpleNamespace(canceled=False,smartsString=query))
    _,report=cycles.select_mcs(elites,previous,cfg,time.time()+30)
    assert report['attempts'][0]['status']=='partial_ring_mapping'
    assert report['selected_elite_count']==1


def test_mcs_logs_previous_elites_extraction_and_reduction(shared_ring_elites,monkeypatch,caplog):
    cfg,previous,elites=shared_ring_elites
    # Attach directly because production logging does not propagate to root.
    cycles.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO,logger=cycles.logger.name):
            cycles.select_mcs(elites,previous,cfg,time.time()+30)
            monkeypatch.setattr(cycles.rdFMCS,'FindMCS',lambda *args:SimpleNamespace(canceled=True,smartsString='invalid partial'))
            cycles.select_mcs(elites,previous,cfg,time.time()+30)
    finally:
        cycles.logger.removeHandler(caplog.handler)
    assert 'MCS previous frame:' in caplog.text
    for row in elites:
        assert f"CMPID_{row['compound_id']} pose={row['id']} MW=" in caplog.text
        assert provenance.canonical_smiles(cycles.restored(row)) in caplog.text
    assert 'frame_SMILES=' in caplog.text and 'MCS SMARTS:' in caplog.text
    assert 'MCS SMARTS readable:' in caplog.text
    assert 'MCS reduce:' in caplog.text and 'discarded_pose_ids=' in caplog.text
    assert 'invalid partial' not in caplog.text


def test_bond_distance_probe_selection_and_only_one_next_frame(project,monkeypatch):
    cfg=project;db,frame,h,payload=seed_completed_cycle(cfg)
    job=database.growth_claim(db,2,2,1)
    assert job['kind']=='frame' and database.growth_claim(db,2,2,2)['kind']=='wait'
    probed=[]
    def probe(mol,history,uid,*args):
        probed.append(uid)
        return dict(status='PASS',reason='synthetic nonclashing probe',pose=None,valid_poses=1)
    monkeypatch.setattr(cycles,'methyl_probe',probe)
    checker=BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
    cycles.prepare_next_frame(db,2,cfg,checker,time.time()+30)
    report=database.growth_cycle_info(db,2)
    assert report['status']=='ready' and report['selected_elite_count']==1
    assert len(probed)==11 # all H-bearing atoms in both newly attached chains
    selected=report['selected_attachments']
    assert [r['bond_distance'] for r in selected]==[6,5,5]
    assert len({r['uid'] for r in selected})==3
    assert all(uid not in h['frame_uids'] for uid in probed)
    assert database.growth_claim(db,2,2,1)['kind']=='plan'
    rows=database.growth_rows(db,'combichem3d_pose')
    assert sum(not r['generated'] for r in rows)==2 # initial MCS and updated MCS, no raw reference ligands
    new=next(r for r in rows if r['iteration']==1 and not r['generated'])
    new_h=json.loads(new['history'])
    assert len(growth_sites(provenance.restore(Chem.MolFromMolBlock(new['molblock']),new_h),new_h))==3
    assert database.growth_structure_rows(db,1,frames_only=True)[0]['pose_id']==new['id']
    export_results(db,cfg['out-dir'])
    assert (Path(cfg['out-dir'])/'cycles.jsonl').is_file()


def test_cycle_preparation_waits_for_upstream_work(project):
    cfg=project;db,_=stage.prepare_project(cfg);database.growth_start_session(db,cfg)
    assert database.growth_claim(db,2,2,1)['kind']=='wait'
    assert database.growth_cycle_info(db,2)=={}


def test_no_elites_stops_without_reusing_reference_ligands(project):
    cfg=project;db,_=stage.prepare_project(cfg);database.growth_start_session(db,cfg)
    database.growth_save_plan(db,1,dict(sites=[],pools=[],total=0),[])
    assert database.growth_claim(db,2,2,1)['kind']=='frame'
    cycles.prepare_next_frame(db,2,cfg,None,time.time()+30)
    assert database.growth_run_info(db)['stop_reason']=='no_eligible_elites'
    assert len(database.growth_rows(db,'compound'))==1


def test_elite_ranking_uses_only_previous_cycle_unique_passing_products(project):
    cfg=project;db,frame,h,payload=seed_completed_cycle(cfg)
    rows=database.growth_elite_rows(db,1,20,'qed')
    assert len(rows)==1 and rows[0]['smiles']==payload['smiles']
    assert database.growth_elite_rows(db,0,20,'qed')==[]
    with pytest.raises(ValueError):database.growth_elite_rows(db,1,20,None)
    with database.growth_connection(db) as conn:
        for number,score in ((1,.95),(2,.95),(3,.1)):
            extra=product_row(frame,h,number,choices=['*'+('C'*(number+2)),'*CCCC'])
            extra['properties']['qed']=score
            database._growth_insert_pose(conn,extra,1)
    ranked=database.growth_elite_rows(db,1,3,'qed')
    assert [r['properties']['qed'] for r in ranked]==[.95,.95,.9]
    assert ranked[0]['compound_id']<ranked[1]['compound_id']


def test_real_two_cycle_search_starts_at_mcs_and_ignores_target_when_enumerating_cycles(project):
    cfg=project
    cfg.update({'target-count':1,'stop-mode':'cycles','max-seconds':40.,'grace-seconds':1.})
    Path(cfg['sidechain-library']).write_text('*C\n*CC\n')
    db,first=stage.prepare_project(cfg)
    stage.supervise(cfg,db,first)
    info=export_results(db,cfg['out-dir'])
    assert info['stop_reason']=='exhausted',info
    attempts=database.growth_rows(db,'combichem3d_attempt')
    assert sum(r['iteration']==1 for r in attempts)==4
    report=database.growth_cycle_info(db,2)
    assert report['status']=='ready'
    assert len(report['selected_attachments'])==2 # only two new H-bearing MCS sites exist
    assert sum(r['iteration']==2 for r in attempts)==4
    assert info['unique_count']>1
    assert len(database.growth_structure_rows(db,0,frames_only=True))==1
    assert len(database.growth_structure_rows(db,1,frames_only=True))==1
    parents=database.growth_rows(db,'combichem3d_pose')
    assert all(json.loads(r['history']).get('role')=='frame' for r in parents if not r['generated'])
    assert all(json.loads(r['history'])['constraint_reference']['record']==1 for r in parents if r['generated'])


def test_prepared_mcs_and_probe_results_survive_interruption(project,monkeypatch):
    cfg=project;db,frame,h,payload=seed_completed_cycle(cfg)
    database.growth_claim(db,2,2,1)
    called=[]
    def probe(mol,history,uid,*args):
        called.append(uid)
        if len(called)==1:database.growth_request_stop(db,'interrupted')
        return dict(status='PASS',reason='synthetic probe',pose=None,valid_poses=1)
    monkeypatch.setattr(cycles,'methyl_probe',probe)
    checker=BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
    cycles.prepare_next_frame(db,2,cfg,checker,time.time()+30)
    saved=database.growth_cycle_info(db,2)
    assert len(saved['probes'])==1 and 'prepared_frame' in saved
    database.growth_end_session(db,'interrupted');database.growth_start_session(db,cfg)
    assert database.growth_claim(db,2,2,1)['kind']=='frame'
    monkeypatch.setattr(cycles,'select_mcs',lambda *args:pytest.fail('Completed MCS must not be recomputed'))
    cycles.prepare_next_frame(db,2,cfg,checker,time.time()+30)
    assert len(called)==len(set(called))==11
    assert database.growth_cycle_info(db,2)['status']=='ready'


@pytest.mark.parametrize('status,reason',[('CLASH_REJECT','no_passing_attachment_points'),('UNRESOLVED','attachment_search_unresolved')])
def test_attachment_failure_states_remain_distinct(project,monkeypatch,status,reason):
    cfg=project;db,_,_,_=seed_completed_cycle(cfg)
    database.growth_claim(db,2,2,1)
    monkeypatch.setattr(cycles,'methyl_probe',lambda *args:dict(status=status,reason='synthetic finite probe result',pose=None))
    cycles.prepare_next_frame(db,2,cfg,None,time.time()+30)
    assert database.growth_run_info(db)['stop_reason']==reason
    assert database.growth_cycle_info(db,2)['status']==reason
    assert len(database.growth_rows(db,'combichem3d_pose'))==2


def test_unresolved_farthest_site_is_not_selected_as_a_passing_attachment(project,monkeypatch):
    cfg=project;db,_,h,payload=seed_completed_cycle(cfg)
    previous=database.growth_previous_frame(db,2)
    template=cycles.restored(payload)
    farthest=cycles.terminal_h_candidates(template,payload['history'],previous)[0][0]
    database.growth_claim(db,2,2,1)
    monkeypatch.setattr(cycles,'methyl_probe',lambda mol,h,uid,*args:
                        dict(status='UNRESOLVED' if uid==farthest else 'PASS',reason='synthetic probe',pose=None))
    checker=BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
    cycles.prepare_next_frame(db,2,cfg,checker,time.time()+30)
    report=database.growth_cycle_info(db,2)
    assert farthest not in [r['uid'] for r in report['selected_attachments']]
    assert [r['bond_distance'] for r in report['selected_attachments']]==[5,5,4]


@pytest.mark.parametrize('smarts,expected,mode',[
    ('c1([*:2])c([*:1])cccc1',{'0':2,'1':1},'map_cycle_limits'),
    ('c1([*:2])c([*:2])cccc1',{'0':2,'1':2},'map_cycle_limits'),
    ('c1([*:999])c([*:999])cccc1',{'0':999,'1':999},'map_cycle_limits'),
    ('c1(*)c(*)cccc1',{'0':999,'1':999},'map_cycle_limits'),
    ('c1([*:0])c([*:1])cccc1',{},'legacy_labels'),
])
def test_dummy_map_numbers_encode_limits_with_independent_root_ids(smarts,expected,mode):
    labels,limits,notation=initial_attachment_spec(Chem.MolFromSmarts(smarts))
    assert labels==['0','1'] and limits==expected and notation==mode
    query,anchors=frames.frame_query(smarts)
    assert query.GetNumAtoms()==6 and len(anchors)==2 and anchors[0]!=anchors[1]


def test_equal_map_cycle_limits_keep_distinct_ancestry_on_resume(project):
    project['reference-mcs']='c1([*:2])c([*:2])cccc1'
    db,_=stage.prepare_project(project)
    previous=database.growth_parent(db,1);h=previous['history']
    assert h['initial_attachment_notation']=='map_cycle_limits'
    assert h['initial_attachment_max_cycles']=={'0':2,'1':2}
    assert h['branch_roots']=={'branch_1':'0','branch_2':'1'}
    assert len(set(h['initial_attachment_labels'].values()))==2
    product=product_row(stage.restore_parent(previous),h)
    assert product['history']['growth_counts']=={'0':1,'1':1}
    assert {root_for_site(product['history'],a['uid']) for a in product['history']['atoms']
            if a['added_iteration']==1}=={'0','1'}
    stage.prepare_project(project)
    assert database.growth_parent(db,1)['history']==h


def test_embedded_map_limits_gate_next_cycle_and_are_shown_in_analysis(project,monkeypatch):
    project['reference-mcs']='c1([*:2])c([*:1])cccc1'
    db,_,_,_=seed_completed_cycle(project);database.growth_claim(db,2,1,1)
    probed=[]
    def probe(mol,h,uid,*args):
        probed.append(root_for_site(h,uid))
        return dict(status='PASS',reason='synthetic probe',pose=None)
    monkeypatch.setattr(cycles,'methyl_probe',probe)
    checker=BumpChecker(load_receptor(project['receptor'],project),project)
    cycles.prepare_next_frame(db,2,project,checker,time.time()+30)
    report=database.growth_cycle_info(db,2)
    assert probed and set(probed)=={'0'}
    assert {a['root_attachment'] for a in report['selected_attachments']}=={'0'}
    assert {a['root_attachment'] for a in report['blocked_attachments']}=={'1'}
    current=database.growth_previous_frame(db,3)
    assert current['history']['growth_counts']=={'0':1,'1':1}
    text='\n'.join(final_analysis_lines(db))
    assert '초기 부착점 표기=map_cycle_limits' in text
    assert 'max_cycles=2 input_map_value=2' in text
    assert 'max_cycles=1 input_map_value=1' in text
    assert 'CYCLE_LIMIT_BLOCKED atom=' in text


@pytest.mark.parametrize('value',['{ "0" = 1, "1" = 2, "3" = 999 }','"0:1, 1:2, 3:999"'])
def test_initial_cycle_limit_config_accepts_table_or_compact_list(project,value):
    path=Path(project['config-path'])
    text=path.read_text().replace('growth-mode = "frontier"','growth-mode = "frame-cycles"')
    path.write_text('initial-attachment-max-cycles = '+value+'\n'+text)
    cfg=combichem_3d_settings(path)
    assert cfg['initial-attachment-max-cycles']=={'0':1,'1':2,'3':999}


@pytest.mark.parametrize('value',['true','{ "0" = true }','{ "0" = 0 }','{ "-1" = 2 }',
                                   '{ "0" = 1.5 }','"0:1, 0:2"','"0:1, 00:2"','"x:2"'])
def test_invalid_initial_cycle_limit_config_is_rejected(project,value):
    path=Path(project['config-path'])
    text=path.read_text().replace('growth-mode = "frontier"','growth-mode = "frame-cycles"')
    path.write_text('initial-attachment-max-cycles = '+value+'\n'+text)
    with pytest.raises(ValueError):combichem_3d_settings(path)


def test_missing_initial_label_is_rejected_before_growth(project):
    project['initial-attachment-max-cycles']={'3':2}
    with pytest.raises(ValueError,match='absent initial attachment labels: 3'):
        stage.prepare_project(project)
    assert not (Path(project['out-dir'])/'combichem3d.sqlite').exists()


def test_root_limits_follow_explicit_labels_and_count_cycles_not_sites(project):
    project['reference-mcs']=project['reference-mcs'].replace('[*:1]','[*:3]')
    project['initial-attachment-max-cycles']={'0':1,'3':999}
    db,_=stage.prepare_project(project);row=database.growth_parent(db,1);frame=stage.restore_parent(row);h=row['history']
    assert set(h['initial_attachment_labels'])=={'0','3'}
    product=product_row(frame,h)
    birth=product['history'];mol=cycles.restored(product)
    assert birth['growth_counts']=={'0':1,'3':1}
    left=[a['uid'] for a in birth['atoms'] if a['branch']=='branch_1']
    right=[a['uid'] for a in birth['atoms'] if a['branch']=='branch_2']
    assert all(root_for_site(birth,uid)=='3' for uid in right)
    with pytest.raises(ValueError,match='attachment 0 reached'):
        assemble(mol,birth,left[:1],['*C'],'blocked')
    _,next_history=assemble(mol,birth,right[:2],['*C','*C'],'two-sites-one-root')
    assert next_history['growth_counts']=={'0':1,'3':2}
    assert birth['growth_counts']=={'0':1,'3':1}
    # A no-op is not a growth cycle for that root.
    _,noop=assemble(frame,h,h['attachments'],['-H','*C'],'noop')
    assert noop['growth_counts']=={'0':0,'3':1}


def test_balancing_covers_both_branches_and_rotates_extra_slot():
    h=dict(initial_attachment_labels={'0':'core0','1':'core1'},branch_roots={'left':'0','right':'1'},
           atoms=[dict(uid=f'{b}{i}',branch=b) for b in ('left','right') for i in range(3)],
           growth_counts={'0':1,'1':1},attachment_selection_counts={'0':0,'1':0})
    passing=[(f'left{i}',9-i) for i in range(3)]+[(f'right{i}',3-i) for i in range(3)]
    selected=balanced_sites(passing,h,3)
    roots=[root_for_site(h,uid) for uid,_ in selected]
    assert roots.count('0')==2 and roots.count('1')==1
    assert {uid for uid,_ in selected}=={'left0','left1','right0'}
    h['attachment_selection_counts']={'0':2,'1':1}
    selected=balanced_sites(passing,h,3)
    roots=[root_for_site(h,uid) for uid,_ in selected]
    assert roots.count('0')==1 and roots.count('1')==2
    h['initial_attachment_max_cycles']={'0':1,'1':2}
    assert all(root_for_site(h,uid)=='1' for uid,_ in balanced_sites(passing,h,3))


def test_balancing_rotates_when_there_are_more_roots_than_slots():
    h=dict(initial_attachment_labels={str(i):f'core{i}' for i in range(4)},
           branch_roots={f'b{i}':str(i) for i in range(4)},
           atoms=[dict(uid=f'a{i}',branch=f'b{i}') for i in range(4)],
           attachment_selection_counts={str(i):0 for i in range(4)})
    passing=[(f'a{i}',10-i) for i in range(4)]
    first=balanced_sites(passing,h,3)
    assert len({root_for_site(h,uid) for uid,_ in first})==3
    for uid,_ in first:
        h['attachment_selection_counts'][root_for_site(h,uid)]+=1
    second=balanced_sites(passing,h,3)
    assert second[0][0]=='a3'
    h['initial_attachment_max_cycles']={'0':999,'1':2}
    h['growth_counts']={'0':10000,'1':2}
    assert growth_available(h,'0') and not growth_available(h,'1')


@pytest.mark.parametrize('notation',['legacy','embedded'])
def test_real_cycle_limits_end_before_global_max_iteration(project,notation):
    project.update({'max-iteration':3,'max-seconds':40.})
    if notation=='legacy':
        project['initial-attachment-max-cycles']={'0':1,'1':1}
    else:
        project['reference-mcs']='c1([*:1])c([*:1])cccc1'
    Path(project['sidechain-library']).write_text('*C\n*CC\n')
    db,first=stage.prepare_project(project);stage.supervise(project,db,first)
    info=export_results(db,project['out-dir'])
    assert info['stop_reason']=='attachment_cycle_limits_reached'
    assert not info['partial_search']
    attempts=database.growth_rows(db,'combichem3d_attempt')
    assert len(attempts)==4 and all(r['iteration']==1 for r in attempts)


def test_limits_skip_probes_and_selection_on_exhausted_branch(project,monkeypatch):
    project['initial-attachment-max-cycles']={'0':1,'1':2}
    db,_,_,_=seed_completed_cycle(project)
    database.growth_claim(db,2,1,1)
    called=[]
    def probe(mol,h,uid,*args):
        called.append(root_for_site(h,uid))
        return dict(status='PASS',reason='synthetic probe',pose=None)
    monkeypatch.setattr(cycles,'methyl_probe',probe)
    checker=BumpChecker(load_receptor(project['receptor'],project),project)
    cycles.prepare_next_frame(db,2,project,checker,time.time()+30)
    report=database.growth_cycle_info(db,2)
    assert called and set(called)=={'1'}
    assert len(report['selected_attachments'])==3
    assert {r['root_attachment'] for r in report['selected_attachments']}=={'1'}
    assert {r['root_attachment'] for r in report['blocked_attachments']}=={'0'}
    frame=database.growth_previous_frame(db,3)
    assert frame['history']['growth_counts']=={'0':1,'1':1}
    assert frame['history']['attachment_selection_counts']=={'0':0,'1':3}
    # Restart retains ancestry/counters and does not re-probe the ready frame.
    database.growth_end_session(db,'interrupted');database.growth_start_session(db,project)
    resumed=database.growth_previous_frame(db,3)
    assert resumed['history']==frame['history']
    assert database.growth_claim(db,2,1,1)['kind']=='plan'


def test_all_root_limits_finish_without_fabricating_a_next_frame(project,monkeypatch):
    project['initial-attachment-max-cycles']={'0':1,'1':1}
    db,_,_,_=seed_completed_cycle(project);database.growth_claim(db,2,1,1)
    monkeypatch.setattr(cycles,'methyl_probe',lambda *args:pytest.fail('Exhausted roots must not be probed'))
    cycles.prepare_next_frame(db,2,project,None,time.time()+30)
    assert database.growth_run_info(db)['stop_reason']=='attachment_cycle_limits_reached'
    assert len(database.growth_rows(db,'combichem3d_pose'))==2
    database.growth_end_session(db,'exhausted')
    assert not export_results(db,project['out-dir'])['partial_search']


def test_attachment_display_preserves_original_and_uses_requested_stars():
    mol=Chem.MolFromSmiles('c1ccccc1CCN')
    h=dict(atoms=[dict(index=a.GetIdx(),uid=f'a{a.GetIdx()}') for a in mol.GetAtoms()])
    original=Chem.MolToSmiles(mol)
    text=attachment_display(mol,h,['a0','a6','a8'])
    assert '[c*:1]' in text and '[CH*:7]' in text and '[NH*:9]' in text
    assert text.count('*')==3
    assert Chem.MolToSmiles(mol)==original and all(a.GetAtomMapNum()==0 for a in mol.GetAtoms())


def test_final_analysis_is_read_only_and_contains_cycle_frames_and_sites(project,monkeypatch):
    db,_,_,_=seed_completed_cycle(project);database.growth_claim(db,2,1,1)
    monkeypatch.setattr(cycles,'methyl_probe',lambda *args:dict(status='PASS',reason='synthetic probe',pose=None))
    checker=BumpChecker(load_receptor(project['receptor'],project),project)
    cycles.prepare_next_frame(db,2,project,checker,time.time()+30)
    before=database.growth_rows(db,'combichem3d_pose')
    report='\n'.join(final_analysis_lines(db))
    assert 'CYCLE 1 FRAME' in report and 'CYCLE 2 FRAME' in report
    assert 'PASS_ATTACH_DISPLAY=' in report and '*:' in report
    assert 'SELECTED_ATTACH_DISPLAY=' in report and 'root=0' in report and 'root=1' in report
    assert 'initial_branch_balanced' in report and 'growth_cycles_used=1' in report
    assert report.endswith('===== 실행 결과 분석 END =====')
    assert database.growth_rows(db,'combichem3d_pose')==before
