"""Explicit heavy-atom provenance; no symmetry-based ancestry inference."""
import json
import math
from rdkit import Chem

SCHEMA = 'molnova_pose_history_v1'
FIELDS = ('uid', 'branch', 'fragment', 'added_iteration')


def tag_atom_cycles(mol, history):
    """Expose saved heavy-atom provenance without guessing hydrogen birth."""
    for row in history['atoms']:
        mol.GetAtomWithIdx(row['index']).SetIntProp('molnova_added_cycle',row['added_iteration'])


def annotate(mol, history):
    mol = Chem.Mol(mol)
    if not isinstance(history,dict):
        raise ValueError('Each parent history must be a JSON object with explicit atom records')
    if history.get('schema') != SCHEMA:
        raise ValueError(f'Atom history must declare schema={SCHEMA}; supply explicit UID/core/branch/fragment/birth records')
    iteration = history.get('iteration')
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError('History iteration must be a nonnegative integer')
    if mol.GetNumConformers() != 1 or not mol.GetConformer().Is3D():
        raise ValueError('Each parent requires exactly one explicit 3D pose in receptor coordinates')
    atoms = history.get('atoms', [])
    if not isinstance(atoms,list) or any(not isinstance(row,dict) for row in atoms):
        raise ValueError('History atoms must be explicit atom record objects')
    if any(isinstance(r.get('index'),bool) or not isinstance(r.get('index'),int) for r in atoms):
        raise ValueError('History atom indices must be zero-based integers')
    if any(not isinstance(row.get(key),str) or not row[key] for row in atoms for key in FIELDS[:3]):
        raise ValueError('Each heavy atom needs nonempty uid, branch and fragment strings')
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if ({r.get('index') for r in atoms} != heavy or len(atoms) != len(heavy)
            or len({r.get('uid') for r in atoms}) != len(atoms)):
        raise ValueError('History must identify every heavy atom index exactly once, with unique UIDs; ancestry will not be inferred')
    fragments = {}
    for row in atoms:
        if any(not isinstance(row.get(k), str) or not row[k] for k in FIELDS[:3]):
            raise ValueError('Each heavy atom needs nonempty uid, branch and fragment strings')
        birth = row.get('added_iteration')
        if isinstance(birth, bool) or not isinstance(birth, int) or not 0 <= birth <= iteration:
            raise ValueError('Every added_iteration must be an integer between zero and parent iteration')
        identity = (row['branch'], birth)
        if fragments.setdefault(row['fragment'], identity) != identity:
            raise ValueError('One fragment must have one branch and added_iteration')
        atom = mol.GetAtomWithIdx(row['index'])
        for key in FIELDS:
            atom.SetProp('_mn_' + key, str(row[key]))
    core = history.get('core', [])
    if not isinstance(core,list) or any(not isinstance(r,dict) for r in core):
        raise ValueError('History core must contain explicit core_index/UID/xyz records')
    if any(not isinstance(r.get('uid'),str) or isinstance(r.get('core_index'),bool)
           or not isinstance(r.get('core_index'),int) for r in core):
        raise ValueError('Core indices must be integers and core UIDs must be strings')
    lookup = {a.GetProp('_mn_uid'): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    query = Chem.MolFromSmarts(history.get('core_smarts', ''))
    if (query is None or not core or len(core) != query.GetNumAtoms()
            or {r.get('core_index') for r in core} != set(range(len(core)))
            or len({r.get('uid') for r in core}) != len(core)):
        raise ValueError('Supply core_smarts and an explicit unique core_index -> atom UID -> xyz mapping')
    mapped = {}
    for row in core:
        xyz = row.get('xyz', [])
        if row.get('uid') not in lookup or len(xyz) != 3 or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in xyz):
            raise ValueError('Core coordinates must be finite xyz in the receptor coordinate frame')
        index = lookup[row['uid']]
        if not query.GetAtomWithIdx(row['core_index']).Match(mol.GetAtomWithIdx(index)):
            raise ValueError('Explicit core atom mapping does not match core_smarts')
        mapped[row['core_index']] = index
    for bond in query.GetBonds():
        actual = mol.GetBondBetweenAtoms(mapped[bond.GetBeginAtomIdx()], mapped[bond.GetEndAtomIdx()])
        if actual is None or not bond.Match(actual):
            raise ValueError('Explicit core bond mapping does not match core_smarts')
    attachments = history.get('attachments', [])
    core_uids = {r['uid'] for r in core}
    attachment_domain=set(history.get('frame_uids',[])) if history.get('growth_mode')=='frame-cycles' else core_uids
    if not attachment_domain<=set(lookup):
        raise ValueError('Every frame UID must identify a retained heavy atom')
    if (not isinstance(attachments,list) or any(not isinstance(uid,str) for uid in attachments)
            or not 1 <= len(attachments) <= 4 or len(set(attachments)) != len(attachments) or not set(attachments) <= attachment_domain):
        raise ValueError('History attachments must list 1–4 distinct core atom UIDs')
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        if not all(math.isfinite(x) for x in pos):
            raise ValueError('Nonfinite parent coordinates')
    # Preserve supplied hydrogen coordinates; they seed independent probe
    # directions and optional all-atom baseline checks. Provenance is heavy-atom.
    clean = Chem.Mol(mol)
    if any(a.GetAtomicNum() == 0 for a in clean.GetAtoms()) or len(Chem.GetMolFrags(clean)) != 1:
        raise ValueError('A parent must be one connected molecule without attachment dummies')
    tag_atom_cycles(clean,history)
    return clean, snapshot(clean, history)


def snapshot(mol, template, iteration=None):
    result = {key: value for key, value in template.items() if key not in ('atoms','pose_coordinates')}
    result['iteration'] = template['iteration'] if iteration is None else iteration
    result['atoms'] = [dict(index=a.GetIdx(), uid=a.GetProp('_mn_uid'), branch=a.GetProp('_mn_branch'),
                            fragment=a.GetProp('_mn_fragment'), added_iteration=int(a.GetProp('_mn_added_iteration')))
                       for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if mol.GetNumConformers():
        result['pose_coordinates'] = [list(mol.GetConformer().GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms()]
    return result


def restore(mol, history):
    for row in history['atoms']:
        for key in FIELDS:
            mol.GetAtomWithIdx(row['index']).SetProp('_mn_' + key, str(row[key]))
    tag_atom_cycles(mol,history)
    coordinates = history.get('pose_coordinates')
    if coordinates is not None:
        if len(coordinates) != mol.GetNumAtoms():
            raise ValueError('Saved pose coordinate count differs from its atom graph')
        for index,xyz in enumerate(coordinates):
            mol.GetConformer().SetAtomPosition(index,xyz)
    return mol


def frontier_uids(history):
    core = {r['uid'] for r in history['core']}
    return [r['uid'] for r in history['atoms'] if r['added_iteration'] == history['iteration'] and r['uid'] not in core]


def canonical_smiles(mol):
    copy = Chem.RemoveHs(Chem.Mol(mol))
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def load_parents(sdf, filename, core_smarts=None):
    with open(filename) as stream:
        document = json.load(stream)
    rows = document.get('parents', []) if isinstance(document,dict) else []
    if not rows:
        raise ValueError('History JSON needs a parents list with explicit atom provenance; symmetric MCS matching is insufficient')
    mols = list(Chem.SDMolSupplier(str(sdf), removeHs=False))
    if len(mols) != len(rows):
        raise ValueError('History parents must correspond one-to-one, in SDF record order')
    result = []
    for index, (mol, row) in enumerate(zip(mols, rows)):
        if mol is None:
            raise ValueError(f'Invalid parent SDF record {index}')
        if isinstance(row,dict):
            row = dict(row,source_name=mol.GetProp('_Name') if mol.HasProp('_Name') else f'parent_{index}')
            if 'core_smarts' not in row and core_smarts is not None:
                row['core_smarts']=core_smarts
        result.append(annotate(mol, row))
    if len({h['iteration'] for _, h in result}) != 1:
        raise ValueError('All imported parents must have the same iteration')
    return result
