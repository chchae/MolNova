"""Bounded constrained conformer search; force-field failures are not clashes."""
import math
import random
import time
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms
from rdkit.Geometry import Point3D
from molnova.states import PoseVerdict
from molnova.chemistry.combichem3d.assembly import assemble, hydrogen_directions, local_methyl_pose


def core_deviation(mol, history):
    lookup = {a.GetProp('_mn_uid'): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    differences = [np.linalg.norm(np.array(mol.GetConformer().GetAtomPosition(lookup[r['uid']])) - r['xyz'])
                   for r in history['core']]
    return float(max(differences))


def fixed_coordinates(product, history, parent=None):
    if parent is not None:
        return {a.GetProp('_mn_uid'): list(parent.GetConformer().GetAtomPosition(a.GetIdx()))
                for a in parent.GetAtoms() if a.GetAtomicNum() > 1}
    return {r['uid']: r['xyz'] for r in history['core']}


def amide_bond(bond):
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    for carbon, nitrogen in ((a, b), (b, a)):
        if carbon.GetAtomicNum() == 6 and nitrogen.GetAtomicNum() == 7 and any(
                other.GetBondType() == Chem.BondType.DOUBLE and other.GetOtherAtom(carbon).GetAtomicNum() in (8, 16)
                for other in carbon.GetBonds()):
            return True
    return False


def torsions(mol, fixed):
    result = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing() or amide_bond(bond):
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(begin).GetAtomicNum() <= 1 or mol.GetAtomWithIdx(end).GetAtomicNum() <= 1:
            continue
        for b, c in ((begin, end), (end, begin)):
            component, stack = set(), [c]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(a.GetIdx() for a in mol.GetAtomWithIdx(current).GetNeighbors()
                             if {current, a.GetIdx()} != {b, c})
            # The axis endpoint c itself does not move. This permits rotating
            # a newly substituted O-H/N-H/C-H direction while both old heavy
            # atoms on the axis remain exactly fixed.
            if (component - {c}) & fixed:
                continue
            left = [a.GetIdx() for a in mol.GetAtomWithIdx(b).GetNeighbors() if a.GetIdx() != c]
            right = [a.GetIdx() for a in mol.GetAtomWithIdx(c).GetNeighbors() if a.GetIdx() != b]
            if left and right:
                result.append((left[0], b, c, right[0]))
                break
    return result


def embedded(product, fixed, seed, cfg, deadline):
    mol = Chem.AddHs(Chem.Mol(product))
    params = AllChem.ETKDGv3()
    if not hasattr(params,'timeout'):
        raise ValueError('This RDKit build lacks bounded embedding timeout support; install the documented RDKit version')
    params.randomSeed = int(seed % 2147483646) + 1
    params.numThreads = 1
    params.maxIterations = 1000
    params.enforceChirality = True
    params.useRandomCoords = bool(seed % 2)
    remaining = math.ceil(max(0, deadline - time.time())) if math.isfinite(deadline) else cfg['embed-timeout']
    params.timeout = max(1, min(cfg['embed-timeout'], remaining))
    mapping = {a.GetIdx(): Point3D(*fixed[a.GetProp('_mn_uid')]) for a in mol.GetAtoms()
               if a.HasProp('_mn_uid') and a.GetProp('_mn_uid') in fixed}
    params.SetCoordMap(mapping)
    if AllChem.EmbedMolecule(mol, params) < 0:
        raise ValueError('Constrained embedding failed')
    reference = Chem.Mol(mol)
    for index, xyz in mapping.items():
        reference.GetConformer().SetAtomPosition(index, xyz)
    rdMolAlign.AlignMol(mol, reference, atomMap=[(i, i) for i in mapping])
    for index, xyz in mapping.items():
        mol.GetConformer().SetAtomPosition(index, xyz)
    return mol


def optimize(mol, fixed, cfg, hard_fixed=False):
    field = cfg['force-field']
    if field in ('auto', 'MMFF') and AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        kind = 'MMFF'
    elif field in ('auto', 'UFF') and AllChem.UFFHasAllMoleculeParams(mol):
        ff = AllChem.UFFGetMoleculeForceField(mol)
        kind = 'UFF'
    else:
        raise ValueError(f'No complete {field} force-field parameters')
    if ff is None:
        raise ValueError('Force-field construction failed')
    mode = 'fixed' if hard_fixed else cfg['core-constraint']
    for atom in mol.GetAtoms():
        if atom.HasProp('_mn_uid') and atom.GetProp('_mn_uid') in fixed:
            index = atom.GetIdx()
            if mode == 'fixed':
                ff.AddFixedPoint(index)
            else:
                method = ff.MMFFAddPositionConstraint if kind == 'MMFF' else ff.UFFAddPositionConstraint
                method(index, 0., cfg['restraint-force'])
    ff.Initialize()
    return_code=ff.Minimize(maxIts=cfg['optimization-iterations'])
    # RDKit code 1 means the iteration limit was reached, not a broken
    # force field. Keep that evidence and still test geometry/stereo/clashes.
    if return_code not in (0,1):
        raise ValueError(f'Force-field optimization failed with return code {return_code}')
    energy = ff.CalcEnergy()
    if not math.isfinite(energy):
        raise ValueError('Nonfinite optimized ligand energy')
    return kind, float(energy),dict(converged=return_code==0,return_code=int(return_code),
                                  status='converged' if return_code==0 else 'iteration_limit',
                                  max_iterations=cfg['optimization-iterations'])


def stereo_valid(mol):
    expected = Chem.Mol(mol)
    Chem.AssignStereochemistry(expected, cleanIt=True, force=True)
    observed = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(observed, replaceExistingTags=True)
    Chem.AssignStereochemistry(observed, cleanIt=True, force=True)
    for a, b in zip(expected.GetAtoms(), observed.GetAtoms()):
        if a.HasProp('_CIPCode') and (not b.HasProp('_CIPCode') or a.GetProp('_CIPCode') != b.GetProp('_CIPCode')):
            return False
    for a, b in zip(expected.GetBonds(), observed.GetBonds()):
        if a.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ) and a.GetStereo() != b.GetStereo():
            return False
    return True


def search(product, history, checker, cfg, seed, *, parent_fixed=None, starts=(), deadline=float('inf')):
    fixed = fixed_coordinates(product, history, parent_fixed)
    rng = random.Random(seed)
    errors, valid, attempts, last = [], set(), 0, None
    optimization_counts={'converged':0,'iteration_limit':0}
    optimization=None
    supplied = list(starts)
    trials = max(cfg['conformers'], len(supplied))
    for trial in range(trials):
        if time.time() >= deadline:
            errors.append('Search time budget reached')
            break
        try:
            with rdBase.BlockLogs():
                base = Chem.Mol(supplied[trial]) if trial < len(supplied) else embedded(
                    product, fixed, rng.randrange(2147483646), cfg, deadline)
            indices = {a.GetIdx() for a in base.GetAtoms() if a.HasProp('_mn_uid') and a.GetProp('_mn_uid') in fixed}
            flexible = torsions(base, indices)
        except (ValueError, RuntimeError) as exc:
            errors.append(str(exc))
            continue
        for torsion_trial in range(1 + cfg['torsion-trials']):
            if time.time() >= deadline:
                errors.append('Search time budget reached')
                break
            if torsion_trial and not flexible:
                break
            candidate = Chem.Mol(base)
            attempts += 1
            try:
                if torsion_trial:
                    # Ring bonds and amides are excluded, and the moved graph
                    # component must not contain a fixed atom.
                    for quartet in flexible:
                        rdMolTransforms.SetDihedralDeg(candidate.GetConformer(), *quartet, rng.uniform(-180, 180))
                with rdBase.BlockLogs():
                    kind, energy, optimization = optimize(candidate, fixed, cfg, hard_fixed=parent_fixed is not None)
                optimization_counts[optimization['status']]+=1
                displacement = core_deviation(candidate, history)
                if displacement > cfg['core-displacement']:
                    raise ValueError(f'Final core displacement {displacement:.6f} exceeds tolerance')
                if parent_fixed is not None and any(
                        np.linalg.norm(np.array(candidate.GetConformer().GetAtomPosition(a.GetIdx())) - fixed[a.GetProp('_mn_uid')]) > 1e-6
                        for a in candidate.GetAtoms() if a.HasProp('_mn_uid') and a.GetProp('_mn_uid') in fixed):
                    raise ValueError('A supposedly fixed parent heavy atom moved')
                if not stereo_valid(candidate):
                    raise ValueError('Specified stereochemistry does not agree with the optimized 3D pose')
                xyz = np.array(candidate.GetConformer().GetPositions())
                if not np.isfinite(xyz).all():
                    raise ValueError('Nonfinite optimized coordinates')
                relevant = [a.GetIdx() for a in candidate.GetAtoms()
                            if cfg['include-hydrogens'] or a.GetAtomicNum()>1]
                key = tuple(np.round(xyz[relevant].reshape(-1), 3))
                valid.add(key)
                last = checker.check(candidate)
                if last['passed']:
                    # Preserve all specified stereo (checked above), and label
                    # newly created stereocentres with this accepted 3D pose.
                    Chem.AssignStereochemistryFrom3D(candidate, replaceExistingTags=False)
                    Chem.AssignStereochemistry(candidate, cleanIt=True, force=True)
                    return dict(status=PoseVerdict.PASS, reason='A valid constrained pose passed clash filtering',
                                pose=Chem.Mol(candidate), valid_poses=len(valid), attempts=attempts,
                                errors=errors[:10], field=kind, ligand_energy=energy,
                                optimization=optimization,optimization_counts=optimization_counts,
                                constraint_scope='parent_heavy_atoms' if parent_fixed is not None else 'original_reference_core',
                                core_max_displacement=displacement, clashes=last)
            except (ValueError, RuntimeError) as exc:
                errors.append(str(exc))
    status = PoseVerdict.CLASH_REJECT if len(valid) >= cfg['minimum-valid-poses'] and not errors else PoseVerdict.UNRESOLVED
    reason = ('All valid poses obtained in the configured finite search clashed; this is not a proof of impossibility'
              if status == PoseVerdict.CLASH_REJECT else 'Insufficient valid poses or embedding/parameter/optimization failure')
    return dict(status=status, reason=reason, pose=None, valid_poses=len(valid), attempts=attempts,
                errors=errors[:10], clashes=last,last_optimization=optimization,
                constraint_scope='parent_heavy_atoms' if parent_fixed is not None else 'original_reference_core',
                optimization_counts=optimization_counts)


def methyl_probe(parent, history, uid, checker, cfg, seed, deadline=float('inf')):
    product, probe_history = assemble(parent, history, [uid], ['*C'], f'probe/{uid}')
    directions = hydrogen_directions(parent, uid)
    starts = [local_methyl_pose(parent, product, uid, direction) for direction in directions]
    if history.get('growth_mode')=='frame-cycles':
        # Only the original iteration-0 core is fixed. Even atoms retained in
        # later elite-MCS frames remain movable alongside the new methyl group.
        result=search(product,probe_history,checker,cfg,seed,starts=starts,deadline=deadline)
        result['phase']='reference_core_'+cfg['core-constraint']
        return result
    first = search(product, probe_history, checker, cfg, seed, parent_fixed=parent, starts=starts, deadline=deadline)
    if first['status'] == PoseVerdict.PASS:
        first['phase'] = 'parent_heavy_fixed'
        return first
    rescue = search(product, probe_history, checker, cfg, seed + 1, deadline=deadline)
    rescue['phase'] = 'reference_core_' + cfg['core-constraint']
    rescue['first_phase'] = {k: v for k, v in first.items() if k != 'pose'}
    # Failures in a first-phase search cannot establish clash rejection even
    # when a subsequent finite search only found clashing poses.
    if rescue['status'] == PoseVerdict.CLASH_REJECT and first['status'] == PoseVerdict.UNRESOLVED:
        rescue['status'] = PoseVerdict.UNRESOLVED
        rescue['reason'] = 'No passing pose; first-phase search contained unresolved failures'
    return rescue
