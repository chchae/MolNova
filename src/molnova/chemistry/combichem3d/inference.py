"""User-authorized approximate import ancestry, never historical reconstruction."""
import json
import os
from pathlib import Path
from rdkit import Chem
from molnova.chemistry.combichem3d.history import SCHEMA, annotate
from molnova.chemistry.combichem3d.logging_utils import logger


def infer_parent(mol, smarts, parent_index=0, iteration=1):
    query=Chem.MolFromSmarts(smarts or '')
    if query is None or query.GetNumAtoms()==0:
        raise ValueError('History inference requires a valid reference-mcs')
    dummies={a.GetIdx() for a in query.GetAtoms() if a.GetAtomicNum()==0}
    anchors=[]
    for index in sorted(dummies):
        atom=query.GetAtomWithIdx(index)
        if atom.GetDegree()!=1 or atom.GetNeighbors()[0].GetIdx() in dummies:
            raise ValueError('reference-mcs attachment dummies must have one retained core neighbor')
        anchors.append(atom.GetNeighbors()[0].GetIdx())
    retained=[a.GetIdx() for a in query.GetAtoms() if a.GetIdx() not in dummies]
    rw=Chem.RWMol(query)
    for index in sorted(dummies,reverse=True):rw.RemoveAtom(index)
    query=rw.GetMol()
    if not query.GetNumAtoms():raise ValueError('reference-mcs has no retained core atoms')
    matches=mol.GetSubstructMatches(query,uniquify=False,useChirality=True,maxMatches=4096)
    if not matches:
        raise ValueError(f'Parent record {parent_index+1} does not match reference-mcs; no core/history was fabricated')
    match=min(matches)
    core=set(match)
    heavy={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()>1}
    outside=heavy-core
    components=[]
    pending=set(outside)
    while pending:
        stack=[min(pending)];component=set()
        while stack:
            index=stack.pop()
            if index not in pending:continue
            pending.remove(index);component.add(index)
            stack.extend(a.GetIdx() for a in mol.GetAtomWithIdx(index).GetNeighbors() if a.GetIdx() in pending)
        components.append(sorted(component))
    boundary=sorted({n.GetIdx() for i in outside for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in core})
    attachments=sorted({match[retained.index(i)] for i in anchors}) if anchors else boundary
    if not 1<=len(attachments)<=4:
        raise ValueError(f'Parent record {parent_index+1}: inferred core needs 1–4 attachment atoms; use mapped dummy sites or explicit history')
    uid=lambda index:f'import:{parent_index}:atom:{index}'
    rows=[dict(index=i,uid=uid(i),branch='core',fragment='core',added_iteration=0) for i in sorted(core)]
    fragments=[]
    for number,component in enumerate(components,1):
        contacts=sorted({n.GetIdx() for i in component for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in core})
        rows.extend(dict(index=i,uid=uid(i),branch=f'inferred_branch_{number}',
                         fragment=f'import:{parent_index}:fragment:{number}',added_iteration=iteration) for i in component)
        fragments.append(dict(atom_indices=component,core_contact_indices=contacts,multiple_core_contacts=len(contacts)>1))
    conf=mol.GetConformer()
    history=dict(schema=SCHEMA,iteration=iteration,core_smarts=Chem.MolToSmarts(query),
                 core=[dict(core_index=i,uid=uid(index),xyz=list(conf.GetAtomPosition(index))) for i,index in enumerate(match)],
                 attachments=[uid(i) for i in attachments],atoms=sorted(rows,key=lambda r:r['index']),
                 provenance=dict(status='INFERRED',method='core match + connected non-core heavy-atom components',
                                 scope='imported ancestry only; subsequent fragment additions have recorded birth iterations',
                                 birth_assumption=f'core=0; every non-core component={iteration}; actual synthesis chronology unknown',
                                 match_selection='lexicographically smallest atom-index tuple among enumerated matches',
                                 enumerated_matches=len(matches),match_limit=4096,match_limit_reached=len(matches)==4096,
                                 ambiguous_core_mapping=len(matches)>1,selected_core_atom_indices=list(match),
                                 fragments=fragments))
    return annotate(mol,history)[1]


def materialize_inferred_history(cfg):
    if cfg.get('history-source')!='inferred':return
    logger.info('inferring import history: parents=%s reference-mcs=%s',cfg['parents'],cfg.get('reference-mcs'))
    mols=list(Chem.SDMolSupplier(cfg['parents'],removeHs=False))
    if not mols or any(m is None for m in mols):
        raise ValueError('Cannot infer history from empty/invalid parent structures')
    document=dict(parents=[infer_parent(m,cfg.get('reference-mcs'),i,cfg['inferred-parent-iteration']) for i,m in enumerate(mols)])
    for index,history in enumerate(document['parents'],1):
        logger.debug('INFERRED parent_record=%s fragments=%s core_matches=%s selected_core=%s',index,
                     len(history['provenance']['fragments']),history['provenance']['enumerated_matches'],
                     history['provenance']['selected_core_atom_indices'])
    text=json.dumps(document,indent=2,sort_keys=True)+'\n'
    destination=Path(cfg['history'])
    if destination.exists():
        if destination.read_text()!=text:
            raise RuntimeError('Inferred history changed; existing history retained. Use a new 3D output directory.')
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.json.tmp');temporary.write_text(text);os.replace(temporary,destination)
    ambiguous=sum(h['provenance']['ambiguous_core_mapping'] for h in document['parents'])
    logger.info('INFERRED history: %s parents, %s ambiguous core mappings; core iteration=0, non-core iteration=%s; actual chronology unknown; %s',
                len(mols),ambiguous,cfg['inferred-parent-iteration'],destination)
