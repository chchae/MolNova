"""Reduce excessive automatic MCS boundaries using the previous two sites."""
import re

from rdkit import Chem

from molnova import _core
from molnova.chemistry.combichem import assemble, _retained_core_query


def _round_trips(inputs, scaffold_smiles):
    """Every selected elite must survive decomposition and exact reassembly."""
    scaffold = Chem.MolFromSmiles(scaffold_smiles)
    for ref in inputs:
        smiles = Chem.MolToSmiles(ref['mol'], isomericSmiles=True)
        pair = _core.decompose_elite_for_libinvent(smiles, scaffold_smiles)
        if pair is None:
            return False
        # assemble orders sites by parsed atom index, as the TL SMILES columns do.
        product = assemble(scaffold, [Chem.MolFromSmiles(s) for s in pair[1].split('|')])
        if product != smiles:
            return False
    return True


def _fragment_scaffold(mol, retained, boundaries, labels):
    """Cut both decorators with RDKit so retained stereochemistry survives."""
    bonds = [mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in boundaries]
    if len(set(bonds)) != 2 or any(mol.GetBondWithIdx(b).GetBondType() != Chem.BondType.SINGLE
                                  or mol.GetBondWithIdx(b).IsInRing() for b in bonds):
        return None
    fragmented = Chem.FragmentOnBonds(mol, bonds,
                                     dummyLabels=[(label + 1, label + 1) for label in labels])
    mappings = []
    fragments = Chem.GetMolFrags(fragmented, asMols=True, fragsMolAtomMapping=mappings)
    for fragment, mapping in zip(fragments, mappings):
        if {idx for idx in mapping if idx < mol.GetNumAtoms()} != retained:
            continue
        for atom in fragment.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(atom.GetIsotope() - 1)
                atom.SetIsotope(0)
        smiles = Chem.MolToSmiles(fragment, isomericSmiles=True)
        return re.sub(r'(?<!\[)\*', '[*:0]', smiles)
    return None


def reduce_to_two_attachments(inputs, inferred_smarts, previous, docking_smarts):
    """Extend previous sites along shared, single-bond, nonbranching prefixes.

    Stop before each branch/ring or the automatic MCS boundary. Never consume
    a complete decorator. If that concrete candidate fails any elite, reuse
    the validated previous scaffold; never drop an elite to fit the core.
    """
    previous_smiles = previous['scaffold_smiles']
    if not _round_trips(inputs, previous_smiles):
        raise RuntimeError('Previous two-attachment scaffold cannot exactly reassemble every selected elite.')
    mol = Chem.RemoveHs(Chem.Mol(inputs[0]['mol']))
    layout = _core._scaffold_attachment_layout(previous_smiles)
    _, sites, labels, dummies, _ = layout
    matched = _core._match_elite_to_scaffold(mol, previous_smiles)
    match = matched[1]
    base = {match[i] for i in range(len(match)) if i not in dummies}
    automatic = Chem.MolFromSmarts(inferred_smarts)
    # Require an embedding that contains the previous retained core. No
    # unconstrained remapping or shrinkage of that established core is allowed.
    embeddings = [set(m) for m in mol.GetSubstructMatches(automatic, useChirality=False)
                  if base <= set(m)]
    candidates = []
    for common in embeddings:
        retained = set(base)
        boundaries = []
        for label in labels:
            info = sites[label]
            anchor, root = match[info['core_qidx']], match[info['dummy_qidx']]
            while root in common:
                atom = mol.GetAtomWithIdx(root)
                following = [n.GetIdx() for n in atom.GetNeighbors()
                             if n.GetIdx() != anchor and n.GetAtomicNum() > 1]
                bond = mol.GetBondBetweenAtoms(anchor, root)
                if (atom.IsInRing() or atom.GetAtomicNum() <= 1 or len(following) != 1
                        or following[0] in retained or bond.GetBondType() != Chem.BondType.SINGLE
                        or mol.GetBondBetweenAtoms(root, following[0]).GetBondType() != Chem.BondType.SINGLE):
                    break
                retained.add(root)
                anchor, root = root, following[0]
            boundaries.append((anchor, root))
        if retained == base:
            continue
        smiles = _fragment_scaffold(mol, retained, boundaries, labels)
        if smiles is not None:
            candidates.append((len(retained), smiles))

    # Prefer the largest verified linear extension; canonical SMILES breaks ties.
    for _, smiles in sorted(set(candidates), key=lambda c: (-c[0], c[1])):
        scaffold = Chem.MolFromSmiles(smiles)
        if (scaffold.HasSubstructMatch(Chem.MolFromSmarts(docking_smarts), useChirality=False)
                and _round_trips(inputs, smiles)):
            return dict(scaffold_smiles=smiles,
                        mcs_smarts=Chem.MolToSmarts(_retained_core_query(scaffold)))
    return None
