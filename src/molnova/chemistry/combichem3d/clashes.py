"""Fixed-receptor vdW overlap filtering with an explicit intraligand policy."""
from itertools import product
import numpy as np
from rdkit import Chem


class SpatialIndex:
    def __init__(self, coordinates, cell=4.):
        self.coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 3))
        self.cell = cell
        self.buckets = {}
        for index, xyz in enumerate(self.coordinates):
            self.buckets.setdefault(tuple(np.floor(xyz / cell).astype(int)), []).append(index)

    def near(self, xyz, radius):
        low, high = np.floor((xyz - radius) / self.cell).astype(int), np.floor((xyz + radius) / self.cell).astype(int)
        return [i for key in product(*(range(a, b + 1) for a, b in zip(low, high))) for i in self.buckets.get(key, ())]


def atom_data(mol, include_hydrogens=False):
    rows = [a for a in mol.GetAtoms() if a.GetAtomicNum() > (0 if include_hydrogens else 1)]
    table = Chem.GetPeriodicTable()
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError('Cannot assign vdW radii to dummy atoms')
    radii = np.array([table.GetRvdw(a.GetAtomicNum()) for a in rows])
    xyz = np.array([list(mol.GetConformer().GetAtomPosition(a.GetIdx())) for a in rows], dtype=float).reshape((-1, 3))
    if not np.isfinite(xyz).all() or not np.isfinite(radii).all() or any(radii <= 0):
        raise ValueError('Nonfinite coordinates or unavailable vdW radius')
    return [a.GetIdx() for a in rows], xyz, radii


class BumpChecker:
    def __init__(self, receptor, cfg):
        self.cfg = cfg
        _, self.xyz, self.radii = atom_data(receptor, cfg['include-hydrogens'])
        if not len(self.xyz):
            raise ValueError('Receptor policy retained no atoms')
        self.index = SpatialIndex(self.xyz)
        self.max_radius = float(self.radii.max())

    def check(self, ligand):
        indices, xyz, radii = atom_data(ligand, self.cfg['include-hydrogens'])
        tolerance = self.cfg['overlap-tolerance']
        receptor_count, internal_count, receptor_max, internal_max = 0, 0, 0., 0.
        examples, internal_examples = [], []
        for i, point in enumerate(xyz):
            near = self.index.near(point, radii[i] + self.max_radius)
            if not near:
                continue
            overlaps = radii[i] + self.radii[near] - np.linalg.norm(self.xyz[near] - point, axis=1)
            receptor_count += int(np.count_nonzero(overlaps > tolerance))
            receptor_max = max(receptor_max, float(overlaps.max()))
            for offset in np.flatnonzero(overlaps > tolerance)[:max(0, 8 - len(examples))]:
                examples.append(dict(ligand_atom=indices[i], receptor_atom=near[offset], overlap=float(overlaps[offset])))
        # BFS only through three bonds; avoid an O(n^2) graph-distance matrix.
        distances = {}
        for index in indices:
            seen, frontier = {index: 0}, [index]
            for depth in range(1, 4):
                following = []
                for current in frontier:
                    for atom in ligand.GetAtomWithIdx(current).GetNeighbors():
                        neighbor = atom.GetIdx()
                        if neighbor not in seen:
                            seen[neighbor] = depth
                            following.append(neighbor)
                frontier = following
            distances[index] = seen
        local = SpatialIndex(xyz)
        max_radius = float(radii.max())
        for i, point in enumerate(xyz):
            for j in local.near(point, radii[i] + max_radius):
                if j <= i:
                    continue
                distance = distances[indices[i]].get(indices[j], 99)
                if distance <= 2 or (distance == 3 and self.cfg['one-four-policy'] == 'exclude'):
                    continue
                scale = self.cfg['one-four-scale'] if distance == 3 and self.cfg['one-four-policy'] == 'scaled' else 1.
                overlap = scale * (radii[i] + radii[j]) - float(np.linalg.norm(point - xyz[j]))
                internal_count += overlap > tolerance
                internal_max = max(internal_max, overlap)
                if overlap > tolerance and len(internal_examples) < 8:
                    internal_examples.append(dict(atom_i=indices[i],atom_j=indices[j],overlap=float(overlap)))
        passed = (receptor_count <= self.cfg['maximum-receptor-clashes']
                  and internal_count <= self.cfg['maximum-internal-clashes'])
        return dict(passed=bool(passed), receptor_clashes=receptor_count, internal_clashes=int(internal_count),
                    receptor_max_overlap=float(receptor_max), internal_max_overlap=float(internal_max),
                    examples=examples, internal_examples=internal_examples)


def load_receptor(path, cfg):
    if str(path).lower().endswith('.pdb'):
        source = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False, proximityBonding=False)
        if source is None or source.GetNumConformers()!=1:
            raise ValueError('Prepared receptor PDB must contain one fixed coordinate model')
        retained = []
        for atom in source.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info is None:
                raise ValueError('PDB receptor is missing residue identity')
            water = info.GetResidueName().strip() in {'HOH', 'WAT', 'H2O', 'DOD'}
            if water and not cfg['include-water']:
                continue
            if info.GetIsHeteroAtom() and not water and not cfg['include-cofactors']:
                continue
            retained.append(atom.GetIdx())
        rw = Chem.RWMol(source)
        retained = set(retained)
        for index in reversed(range(source.GetNumAtoms())):
            if index not in retained:
                rw.RemoveAtom(index)
        receptor = rw.GetMol()
    elif str(path).lower().endswith('.sdf'):
        if not cfg['include-cofactors']:
            raise ValueError('Cofactor exclusion requires residue-labelled PDB; SDF components are retained explicitly')
        records = list(Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False))
        if not records or any(m is None or m.GetNumConformers() != 1 for m in records):
            raise ValueError('Receptor SDF must contain valid coordinate records')
        receptor = None
        for mol in records:
            # SDF has no standard residue classification. Require an explicit
            # component tag rather than guessing that an oxygen/ion is water.
            is_water = mol.HasProp('molnova_component') and mol.GetProp('molnova_component') == 'water'
            if is_water and not cfg['include-water']:
                continue
            receptor = mol if receptor is None else Chem.CombineMols(receptor, mol)
        if receptor is None:
            raise ValueError('Water/cofactor policy retained no receptor atoms')
    else:
        raise ValueError('Prepared receptor must be PDB or SDF in the parent coordinate frame')
    return receptor
