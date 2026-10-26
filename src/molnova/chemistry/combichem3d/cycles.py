"""Elite-MCS frame updates, calculated only by a claimed independent worker."""
import copy
import hashlib
import math
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFMCS
from molnova import database
from molnova.chemistry.combichem3d import history as provenance
from molnova.chemistry.combichem3d.frames import trim_to_uids
from molnova.chemistry.combichem3d.search import methyl_probe
from molnova.chemistry.combichem3d.logging_utils import logger
from molnova.chemistry.scaffold import mcs_readable_smarts
from molnova.chemistry.combichem3d.attachment_policy import root_for_site, growth_available, balanced_sites


def restored(row):
    mol=Chem.MolFromMolBlock(row['molblock'],removeHs=False)
    if mol is None:raise ValueError('Invalid saved elite/frame pose')
    return provenance.restore(mol,row['history'])


def mcs_input(mol):
    """Normalize graph traversal for FMCS without changing saved stereo/poses.

    FMCS chirality matching can miss a shared ring in SDF-derived bond orders.
    Isomeric SMILES reconstruction preserves chemical identity and stereo; only
    these graph-only copies enter FMCS. Extraction always uses the original pose.
    """
    return Chem.MolFromSmiles(provenance.canonical_smiles(mol))


def cuts_ring(mol, mapping):
    kept=set(mapping)
    return any(b.IsInRing() and ((b.GetBeginAtomIdx() in kept)!=(b.GetEndAtomIdx() in kept))
               for b in mol.GetBonds())


def growth_site_coverage(frame, history, previous):
    """Require a future H-bearing site for every still-active limited root.

    Empty legacy limit metadata retains its original uncapped-MCS behavior.
    New map-cycle histories include explicit limits, including unlimited 999.
    This is a structural check; methyl probes still decide clash eligibility.
    """
    required=([root for root in history.get('initial_attachment_labels',{}) if growth_available(history,root)]
              if history.get('initial_attachment_max_cycles') else [])
    counts={root:0 for root in required}
    if required:
        for uid,_ in terminal_h_candidates(frame,history,previous):
            root=root_for_site(history,uid)
            if root in counts:
                counts[root]+=1
    return dict(required_growth_roots=required,growth_candidate_counts=counts,
                missing_growth_roots=[root for root in required if not counts[root]])


def select_mcs(elites, previous, cfg, deadline, checkpoint=lambda report:None):
    old=set(previous['history']['frame_uids'])
    old_smiles=previous['history']['frame_smiles']
    count=min(len(elites),cfg['elite-count'])
    subset=elites[:count]
    molecular_weights={r['id']:Descriptors.MolWt(Chem.RemoveHs(restored(r))) for r in subset}
    cycle=previous['history']['iteration']+2
    logger.info('MCS previous frame: cycle=%s CMPID_%s pose=%s SMILES=%s SMARTS=%s',
                cycle,previous['compound_id'],previous['id'],old_smiles,previous['history'].get('frame_smarts',''))
    attempts=[]
    while count:
        if time.time()>=deadline:return None,dict(attempts=attempts,status='budget_interrupted')
        template=restored(subset[0])
        attempt=dict(elite_count=count,elite_pose_ids=[r['id'] for r in subset],rank_by=cfg['elite-rank-by'],
                     reduction_by='molecular_weight',molecular_weights=[molecular_weights[r['id']] for r in subset],
                     elite_smiles=[provenance.canonical_smiles(restored(r)) for r in subset],
                     previous_frame_smiles=old_smiles)
        logger.info('MCS attempt: cycle=%s elites=%s initial_rank_by=%s reduction_by=molecular_weight template_pose=%s',
                    cycle,count,cfg['elite-rank-by'],subset[0]['id'])
        for row,smiles in zip(subset,attempt['elite_smiles']):
            logger.info('MCS elite: cycle=%s elites=%s CMPID_%s pose=%s MW=%.3f SMILES=%s',
                        cycle,count,row['compound_id'],row['id'],molecular_weights[row['id']],smiles)
        if count==1:
            frame=Chem.RemoveHs(Chem.Mol(template))
            attempt.update(status='single_elite_full_structure',mcs_smarts=Chem.MolToSmarts(frame))
        else:
            params=rdFMCS.MCSParameters()
            params.Timeout=max(1,min(cfg['mcs-timeout'],math.ceil(deadline-time.time())))
            params.AtomCompareParameters.RingMatchesRingOnly=True
            params.AtomCompareParameters.MatchFormalCharge=True
            params.AtomCompareParameters.MatchChiralTag=True
            params.AtomCompareParameters.CompleteRingsOnly=True
            params.BondCompareParameters.RingMatchesRingOnly=True
            params.BondCompareParameters.CompleteRingsOnly=True
            params.InitialSeed=old_smiles
            molecules=[Chem.RemoveHs(restored(r)) for r in subset]
            result=rdFMCS.FindMCS([mcs_input(m) for m in molecules],params)
            if result.canceled:
                attempt.update(status='mcs_timeout',mcs_smarts=None,timeout_seconds=params.Timeout)
                frame=None
            else:
                query=Chem.MolFromSmarts(result.smartsString)
                if query is None or not query.GetNumAtoms():raise ValueError('RDKit returned an empty elite MCS')
                mappings=template.GetSubstructMatches(query,uniquify=False,useChirality=True,maxMatches=4096)
                attempt.update(template_matches_enumerated=len(mappings),template_match_limit_reached=len(mappings)==4096)
                mappings=[m for m in mappings if old<={template.GetAtomWithIdx(i).GetProp('_mn_uid') for i in m}]
                if not mappings:
                    frame=None;attempt.update(status='previous_frame_not_preserved',mcs_smarts=result.smartsString)
                else:
                    mappings=[m for m in mappings if not cuts_ring(template,m)]
                    if not mappings:
                        frame=None;attempt.update(status='partial_ring_mapping',mcs_smarts=result.smartsString)
                    else:
                        frame=None
                        attempt.update(status='frame_stereo_not_shared',mcs_smarts=result.smartsString)
                        # A degenerate mapping can retain a different branch.
                        # Try every bounded, ring-preserving mapping before
                        # rejecting the MCS for missing active-root growth sites.
                        for mapping in sorted(mappings):
                            candidate=trim_to_uids(template,{template.GetAtomWithIdx(i).GetProp('_mn_uid') for i in mapping})
                            if not all(m.HasSubstructMatch(candidate,useChirality=True) for m in molecules):
                                continue
                            coverage=growth_site_coverage(candidate,subset[0]['history'],previous)
                            attempt.update(coverage,selected_template_indices=list(mapping),
                                           frame_smiles=provenance.canonical_smiles(candidate),atoms=candidate.GetNumHeavyAtoms())
                            if coverage['missing_growth_roots'] and attempt['frame_smiles']!=old_smiles:
                                attempt['status']='missing_active_root_sites'
                                continue
                            frame=candidate;attempt['status']='accepted'
                            break
        if frame is not None:
            smiles=provenance.canonical_smiles(frame)
            attempt.update(frame_smiles=smiles,atoms=frame.GetNumHeavyAtoms())
            if count>1 and smiles==old_smiles:
                attempt['status']='same_as_previous_frame';frame=None
            else:
                coverage=growth_site_coverage(frame,subset[0]['history'],previous)
                attempt.update(coverage)
                if coverage['missing_growth_roots']:
                    attempt['status']='missing_active_root_sites';frame=None
        attempts.append(attempt)
        logger.info('elite MCS: cycle=%s elites=%s status=%s atoms=%s frame_SMILES=%s',
                    cycle,count,attempt['status'],attempt.get('atoms'),attempt.get('frame_smiles'))
        if attempt.get('missing_growth_roots'):
            logger.info('MCS growth requirements: cycle=%s elites=%s required_roots=%s candidate_counts=%s missing_roots=%s',
                        cycle,count,attempt['required_growth_roots'],attempt['growth_candidate_counts'],attempt['missing_growth_roots'])
        if attempt.get('mcs_smarts'):
            logger.info('MCS SMARTS: %s',attempt['mcs_smarts'])
            logger.info('MCS SMARTS readable: %s',mcs_readable_smarts(attempt['mcs_smarts']))
        checkpoint(dict(attempts=attempts))
        if frame is not None:
            history=provenance.snapshot(frame,subset[0]['history'])
            history.update(role='frame',frame_uids=[a.GetProp('_mn_uid') for a in frame.GetAtoms()],
                           frame_smarts=attempt['mcs_smarts'],frame_smiles=provenance.canonical_smiles(frame),
                           previous_attachments=previous['history']['attachments'],attachments=previous['history']['attachments'])
            return (frame,history),dict(attempts=attempts,template_pose=subset[0]['id'],
                                      elite_pose_ids=[r['id'] for r in subset],selected_elite_count=count)
        if count==1:
            return None,dict(attempts=attempts,status='required_growth_sites_unavailable',
                             missing_growth_roots=attempt.get('missing_growth_roots',[]))
        count=max(1,count//2)
        # Stable ties retain the original elite ranking. Only reduction changes
        # the subset order; the first MCS still uses the configured rank order.
        reduced=sorted(subset,key=lambda r:-molecular_weights[r['id']])[:count]
        logger.info('MCS reduce: cycle=%s reason=%s elites=%s->%s kept_pose_ids=%s discarded_pose_ids=%s',
                    cycle,attempt['status'],len(subset),count,[r['id'] for r in reduced],
                    [r['id'] for r in subset if r not in reduced])
        subset=reduced
    raise ValueError('No elite frame available')


def terminal_h_candidates(frame, history, previous):
    old=set(previous['history']['frame_uids'])
    lookup={a.GetProp('_mn_uid'):a.GetIdx() for a in frame.GetAtoms() if a.GetAtomicNum()>1}
    anchors=[lookup[uid] for uid in previous['history']['attachments']]
    distances=Chem.GetDistanceMatrix(frame)
    # A terminal H defines a substitutable direction, also on ring C-H/N-H.
    # Keep one logical candidate per heavy atom, even for CH3/NH2.
    return sorted([(a.GetProp('_mn_uid'),int(min(distances[a.GetIdx(),j] for j in anchors)))
                   for a in frame.GetAtoms() if a.GetAtomicNum()>1 and a.GetProp('_mn_uid') not in old
                   and a.GetTotalNumHs(includeNeighbors=True)>0],key=lambda row:(-row[1],row[0]))


def prepare_next_frame(db, iteration, cfg, checker, deadline):
    previous=database.growth_previous_frame(db,iteration)
    saved=database.growth_cycle_info(db,iteration)
    if 'prepared_frame' in saved:
        frame=Chem.MolFromMolBlock(saved['prepared_frame']['molblock'],removeHs=False)
        history=saved['prepared_frame']['history']
        frame=provenance.restore(frame,history)
        report=saved
    else:
        previous_attempts=list(saved.get('previous_mcs_attempts',[]))
        if saved.get('attempts'):previous_attempts.append(saved['attempts'])
        elites=database.growth_elite_rows(db,iteration-1,cfg['elite-count'],cfg['elite-rank-by'])
        logger.info('cycle %s elite selection: previous_cycle=%s requested=%s available_selected=%s rank_by=%s',
                    iteration,iteration-1,cfg['elite-count'],len(elites),cfg['elite-rank-by'])
        if not elites:
            database.growth_cycle_checkpoint(db,iteration,dict(status='no_eligible_elites'))
            database.growth_request_stop(db,'no_eligible_elites')
            return
        def checkpoint(value):database.growth_cycle_checkpoint(db,iteration,dict(value,previous_mcs_attempts=previous_attempts))
        selected,report=select_mcs(elites,previous,cfg,deadline,checkpoint)
        if selected is None:
            if report.get('status')=='required_growth_sites_unavailable':
                checkpoint(report)
                database.growth_request_stop(db,report['status'])
            return
        frame,history=selected
        report['previous_mcs_attempts']=previous_attempts
        report.update(prepared_frame=dict(molblock=Chem.MolToMolBlock(frame),history=provenance.snapshot(frame,history)),
                      probes={},distance_metric='minimum shortest bond distance to any previous attachment atom')
        checkpoint(report)
    passing=[]
    report['selection_policy']='initial_branch_balanced'
    report['blocked_attachments']=[]
    candidates=terminal_h_candidates(frame,history,previous)
    for uid,distance in candidates:
        if database.growth_should_stop(db):return
        root=root_for_site(history,uid)
        if not growth_available(history,root):
            report['blocked_attachments'].append(dict(uid=uid,root_attachment=root,bond_distance=distance,
                reason='cycle_limit',growth_cycles_used=history['growth_counts'].get(root,0),
                max_cycles=history.get('initial_attachment_max_cycles',{}).get(root,999)))
            continue
        probe=report['probes'].get(uid)
        if probe is None or probe.get('budget_interrupted'):
            seed=int.from_bytes(hashlib.sha256(f'{cfg["seed"]}:{iteration}:{uid}'.encode()).digest()[:4],'big')%2147483646
            result=methyl_probe(frame,history,uid,checker,cfg,seed,deadline)
            probe={k:v for k,v in result.items() if k!='pose'}
            probe.update(bond_distance=distance,budget_interrupted=time.time()>=deadline,gates_enumeration=False,
                         root_attachment=root)
            report['probes'][uid]=probe
            database.growth_cycle_checkpoint(db,iteration,report)
        if probe['status']=='PASS':passing.append((uid,distance))
    passing.sort(key=lambda row:(-row[1],row[0]))
    selected=balanced_sites(passing,history,cfg['next-attachment-count'])
    if not selected:
        unresolved=any(p['status']=='UNRESOLVED' for p in report['probes'].values())
        limited=(bool(candidates) and len(report['blocked_attachments'])==len(candidates)) or all(
            not growth_available(history,r) for r in history['initial_attachment_labels'])
        report['status']=('attachment_cycle_limits_reached' if limited else
                          'attachment_search_unresolved' if unresolved else 'no_passing_attachment_points')
        database.growth_cycle_checkpoint(db,iteration,report)
        database.growth_request_stop(db,report['status'])
        return
    history=copy.deepcopy(history)
    history['attachments']=[uid for uid,_ in selected]
    for uid,_ in selected:
        root=root_for_site(history,uid)
        counts=history.setdefault('attachment_selection_counts',{})
        counts[root]=counts.get(root,0)+1
    frame,history=provenance.annotate(frame,history)
    report.update(status='ready',selected_attachments=[dict(uid=uid,bond_distance=d,
                  root_attachment=root_for_site(history,uid)) for uid,d in selected])
    baseline=checker.check(Chem.AddHs(frame,addCoords=True))
    report['frame_baseline']=baseline
    payload=dict(smiles=provenance.canonical_smiles(frame),molblock=Chem.MolToMolBlock(frame),history=history,
                 properties={},output_passed=False)
    pose=database.growth_save_cycle_frame(db,iteration,payload,report,{uid:report['probes'][uid] for uid,_ in selected})
    row=database.growth_parent(db,pose)
    logger.info('FRAME cycle=%s CMPID_%s pose=%s template_pose=%s SMILES=%s',
                iteration,row['compound_id'],pose,report['template_pose'],payload['smiles'])
    logger.info('cycle %s frame ready: pose=%s SMILES=%s probe_PASS=%s selected_points=%s bond_distances=%s',
                iteration,pose,payload['smiles'],len(passing),len(selected),[d for _,d in selected])
