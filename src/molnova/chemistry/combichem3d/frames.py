"""Frame extraction and pose references; reference ligands never seed growth."""
import copy
import json
import os
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from molnova.chemistry.combichem3d import history as provenance
from molnova.chemistry.combichem3d.logging_utils import logger
from molnova.chemistry.combichem3d.attachment_policy import initial_attachment_spec


def frame_query(smarts):
    query=Chem.MolFromSmarts(smarts or '')
    if query is None or not query.GetNumAtoms():
        raise ValueError('frame-cycles requires reference-mcs with 1–4 terminal attachment dummies')
    dummies=[a.GetIdx() for a in query.GetAtoms() if a.GetAtomicNum()==0]
    anchors=[]
    for index in dummies:
        atom=query.GetAtomWithIdx(index)
        if atom.GetDegree()!=1 or atom.GetNeighbors()[0].GetAtomicNum()==0:
            raise ValueError('Each attachment dummy needs one retained core neighbor')
        anchors.append(atom.GetNeighbors()[0].GetIdx())
    kept=[a.GetIdx() for a in query.GetAtoms() if a.GetIdx() not in dummies]
    anchors=[kept.index(i) for i in anchors]
    if not 1<=len(set(anchors))<=4 or len(anchors)!=len(set(anchors)):
        raise ValueError('reference-mcs must specify 1–4 distinct core attachment atoms')
    rw=Chem.RWMol(query)
    for i in sorted(dummies,reverse=True):rw.RemoveAtom(i)
    return rw.GetMol(),anchors


def trim_to_uids(mol, keep):
    """Retain the template subgraph/coordinates and cap cut valences with H."""
    mol=Chem.RemoveHs(Chem.Mol(mol))
    rw=Chem.RWMol(mol)
    removed=[]
    for atom in rw.GetAtoms():
        if atom.GetProp('_mn_uid') not in keep:
            removed.append(atom.GetIdx())
        elif any(n.GetProp('_mn_uid') not in keep for n in atom.GetNeighbors()):
            atom.SetNoImplicit(False)
    for index in sorted(removed,reverse=True):rw.RemoveAtom(index)
    result=rw.GetMol()
    Chem.SanitizeMol(result)
    Chem.AssignStereochemistry(result,cleanIt=True,force=True)
    if len(Chem.GetMolFrags(result))!=1:
        raise ValueError('MCS frame must be connected')
    return result


class ReferencePoses:
    def __init__(self, filename, smarts):
        self.query,_=frame_query(smarts)
        self.generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
        self.records=[]
        self.match_counts={}
        for index,mol in enumerate(Chem.SDMolSupplier(str(filename),removeHs=False),1):
            if mol is None or mol.GetNumConformers()!=1 or not mol.GetConformer().Is3D():
                raise ValueError(f'Invalid 3D reference-pose record {index}')
            matches=mol.GetSubstructMatches(self.query,uniquify=False,useChirality=True,maxMatches=4096)
            if matches:
                self.match_counts[index]=len(matches)
                self.records.append((index,mol,min(matches),self.generator.GetFingerprint(Chem.RemoveHs(mol))))
        if not self.records:raise ValueError('No reference pose matches the retained reference-mcs')

    def constrain(self, product, history):
        fp=self.generator.GetFingerprint(Chem.RemoveHs(product))
        scores=DataStructs.BulkTanimotoSimilarity(fp,[r[3] for r in self.records])
        selected=max(range(len(scores)),key=lambda i:(scores[i],-self.records[i][0]))
        record,mol,match,_=self.records[selected]
        result=copy.deepcopy(history)
        for row in result['core']:
            row['xyz']=list(mol.GetConformer().GetAtomPosition(match[row['core_index']]))
        result['constraint_reference']=dict(record=record,title=mol.GetProp('_Name') if mol.HasProp('_Name') else '',
                                            similarity=float(scores[selected]),metric='Morgan radius 2 / 2048 Tanimoto',
                                            selected_atom_indices=list(match),enumerated_matches=self.match_counts[record],
                                            match_limit_reached=self.match_counts[record]==4096)
        return result


def initial_frame(filename, smarts, *, legacy_limits=None):
    references=ReferencePoses(filename,smarts)
    record,source,match,_=references.records[0]
    query,anchors=frame_query(smarts)
    source_query=Chem.MolFromSmarts(smarts)
    labels,limits,notation=initial_attachment_spec(source_query,legacy_limits)
    mol=Chem.Mol(source)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()>1:
            atom.SetProp('_mn_uid',f'initial:core:{atom.GetIdx()}')
            atom.SetProp('_mn_branch','core');atom.SetProp('_mn_fragment','core');atom.SetProp('_mn_added_iteration','0')
    uids=[mol.GetAtomWithIdx(i).GetProp('_mn_uid') for i in match]
    frame=trim_to_uids(mol,set(uids))
    lookup={a.GetProp('_mn_uid'):a.GetIdx() for a in frame.GetAtoms()}
    history=dict(schema=provenance.SCHEMA,iteration=0,core_smarts=Chem.MolToSmarts(query),
                 core=[dict(core_index=i,uid=uid,xyz=list(frame.GetConformer().GetAtomPosition(lookup[uid]))) for i,uid in enumerate(uids)],
                 attachments=[uids[i] for i in anchors],atoms=[],role='frame',growth_mode='frame-cycles',
                 frame_uids=[a.GetProp('_mn_uid') for a in frame.GetAtoms()],frame_smarts=Chem.MolToSmarts(frame),
                 frame_smiles=provenance.canonical_smiles(frame),reference_record=record,
                 initial_attachment_labels={label:uids[i] for label,i in zip(labels,anchors)},
                 initial_attachment_max_cycles=limits,
                 initial_attachment_notation=notation,
                 initial_attachment_map_values={label:a.GetAtomMapNum() if a.HasProp('molAtomMapNumber') else None
                    for label,a in zip(labels,[a for a in source_query.GetAtoms() if a.GetAtomicNum()==0])},
                 branch_roots={f'branch_{i+1}':label for i,label in enumerate(labels)},
                 growth_counts={label:0 for label in labels},
                 attachment_selection_counts={label:0 for label in labels},
                 provenance=dict(status='MAPPED_REFERENCE_CORE',note='Only extracted MCS seeds growth; external reference substituents excluded'))
    history=provenance.snapshot(frame,history)
    return provenance.annotate(frame,history)


def prepare_frame_inputs(cfg):
    cfg['reference-poses']=cfg.get('reference-poses',cfg['parents'])
    mol,history=initial_frame(cfg['reference-poses'],cfg.get('reference-mcs'),
                              legacy_limits=cfg.get('initial-attachment-max-cycles'))
    folder=Path(cfg['out-dir'])/'input-cache/frame-seed'
    folder.mkdir(parents=True,exist_ok=True)
    sdf=Chem.MolToMolBlock(mol)+'\n$$$$\n'
    files={'frame.sdf':sdf,'history.json':json.dumps(dict(parents=[history]),indent=2,sort_keys=True)+'\n'}
    for name,text in files.items():
        path=folder/name
        if path.exists() and path.read_text()!=text:
            raise RuntimeError('Initial frame changed; retain old output and use a new output directory')
        if not path.exists():
            temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(text);os.replace(temp,path)
    cfg['parents']=str(folder/'frame.sdf');cfg['history']=str(folder/'history.json');cfg['history-source']='frame-seed'
    logger.info('initial MCS frame: SMILES=%s attachment_points=%s; reference ligands supply coordinates only',history['frame_smiles'],len(history['attachments']))
    logger.info('initial attachment limits: notation=%s roots=%s max_cycles=%s',
                history['initial_attachment_notation'],history['initial_attachment_labels'],history['initial_attachment_max_cycles'])
