"""Replace one H per selected heavy atom; retain all fragment birth records."""
import hashlib
import numpy as np
from rdkit import Chem
from molnova.chemistry.combichem3d.history import frontier_uids, snapshot, tag_atom_cycles
from molnova.chemistry.combichem3d.attachment_policy import root_for_site, growth_available


def growth_sites(parent, history):
    permitted = set(history['attachments'] if history['iteration'] == 0 or history.get('growth_mode')=='frame-cycles' else frontier_uids(history))
    # Do not collapse symmetry-equivalent sites: multiple simultaneous
    # substitutions can break that symmetry in different ways.
    return [a.GetProp('_mn_uid') for a in parent.GetAtoms()
            if a.GetAtomicNum() > 1 and a.GetProp('_mn_uid') in permitted
            and a.GetTotalNumHs(includeNeighbors=True) > 0]


def hydrogen_directions(parent, uid):
    mol = Chem.AddHs(parent, addCoords=True)
    anchor = next(a for a in mol.GetAtoms() if a.HasProp('_mn_uid') and a.GetProp('_mn_uid') == uid)
    origin = np.array(mol.GetConformer().GetAtomPosition(anchor.GetIdx()))
    result = []
    for neighbor in anchor.GetNeighbors():
        if neighbor.GetAtomicNum() == 1:
            delta = np.array(mol.GetConformer().GetAtomPosition(neighbor.GetIdx())) - origin
            norm = np.linalg.norm(delta)
            if norm > 1e-8:
                result.append(delta / norm)
    return result


def assemble(parent, history, sites, choices, token):
    if len(sites) != len(choices) or len(sites) != len(set(sites)):
        raise ValueError('Exactly one choice per distinct growth-site UID is required')
    mol = Chem.AddHs(Chem.Mol(parent), addCoords=parent.GetNumConformers() > 0)
    lookup = {a.GetProp('_mn_uid'): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    iteration = history['iteration'] + 1
    lineage = hashlib.sha256((history.get('lineage_id','') + '/' + token + '/' + repr(list(zip(sites,choices)))).encode()).hexdigest()
    fragments = []
    grown_roots=set()
    label = max((a.GetAtomMapNum() for a in mol.GetAtoms()), default=0) + 1
    rw = Chem.RWMol(mol)
    for site, choice in zip(sites, choices):
        if site not in lookup:
            raise ValueError(f'Unknown growth atom UID: {site}')
        anchor = rw.GetAtomWithIdx(lookup[site])
        hs = [a.GetIdx() for a in anchor.GetNeighbors() if a.GetAtomicNum() == 1]
        if not hs:
            raise ValueError(f'Growth atom {site} has no replaceable H')
        if choice == '-H':
            continue
        if history.get('initial_attachment_labels'):
            root=root_for_site(history,site)
            if not growth_available(history,root):
                raise ValueError(f'Initial attachment {root} reached its cycle limit')
            grown_roots.add(root)
        fragment = Chem.MolFromSmiles(choice)
        dummies = [a for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) != 1 or dummies[0].GetDegree() != 1 or dummies[0].GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
            raise ValueError('Expected a validated single-attachment fragment')
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(label)
        # Replace the explicit H at its neighbor position instead of deleting
        # the terminal heavy atom. molzip preserves attachment stereochemistry.
        rw.ReplaceAtom(hs[0], dummy)
        branch = (f"branch_{history['attachments'].index(site) + 1}"
                  if history['iteration'] == 0 else anchor.GetProp('_mn_branch'))
        fragment_id = hashlib.sha256(f'{lineage}/{site}/{choice}'.encode()).hexdigest()[:24]
        for atom in fragment.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.GetAtomicNum() > 1:
                atom.SetProp('_mn_uid', f'{fragment_id}:{atom.GetIdx()}')
                atom.SetProp('_mn_branch', branch)
                atom.SetProp('_mn_fragment', fragment_id)
                atom.SetProp('_mn_added_iteration', str(iteration))
        dummies[0].SetAtomMapNum(label)
        fragments.append(fragment)
        label += 1
    product = rw.GetMol()
    for fragment in fragments:
        product = Chem.CombineMols(product, fragment)
    if fragments:
        product = Chem.molzip(product)
    Chem.SanitizeMol(product)
    product = Chem.RemoveHs(product)
    product.RemoveAllConformers()
    if any(a.GetAtomicNum() == 0 for a in product.GetAtoms()):
        raise ValueError('Unconsumed attachment dummy after assembly')
    result = snapshot(product, history, iteration)
    tag_atom_cycles(product,result)
    if history.get('growth_mode')=='frame-cycles':
        result['role']='product'
    if history.get('initial_attachment_labels'):
        result['growth_counts']=dict(history['growth_counts'])
        for root in grown_roots:
            result['growth_counts'][root]+=1
    result['parent_lineage'] = history.get('lineage_id')
    result['lineage_id'] = lineage
    result['parent_choices'] = dict(zip(sites, choices))
    return product, result


def local_methyl_pose(parent, product, site, direction):
    """Independent probe starting pose, with every parent heavy atom unchanged."""
    pose = Chem.Mol(product)
    old = {a.GetProp('_mn_uid'): a.GetIdx() for a in parent.GetAtoms() if a.GetAtomicNum() > 1}
    conf = Chem.Conformer(pose.GetNumAtoms())
    conf.Set3D(True)
    anchor = np.array(parent.GetConformer().GetAtomPosition(old[site]))
    for atom in pose.GetAtoms():
        uid = atom.GetProp('_mn_uid')
        xyz = (parent.GetConformer().GetAtomPosition(old[uid]) if uid in old
               else anchor + 1.5 * np.asarray(direction))
        conf.SetAtomPosition(atom.GetIdx(), xyz)
    pose.RemoveAllConformers()
    pose.AddConformer(conf)
    return Chem.AddHs(pose, addCoords=True)
