"""RDKit properties of original generated SMILES, before structure preparation."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import math
import os

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors

from molnova.chemistry.sa_score import calculate_sa_score_for_mol


@lru_cache(maxsize=2)
def _catalog(name):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog.FilterCatalog(params)


def calculate_rdkit_descriptors(smiles, sa_score=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f'Cannot calculate RDKit properties for invalid or empty SMILES: {smiles!r}')
    sa = calculate_sa_score_for_mol(mol) if sa_score is None else float(sa_score)
    if not math.isfinite(sa) or not 1 <= sa <= 10:
        raise ValueError(f'Invalid existing SA score: {sa}')
    mw = float(Descriptors.MolWt(mol))
    clogp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotatable = int(rdMolDescriptors.CalcNumRotatableBonds(
        mol, rdMolDescriptors.NumRotatableBondsOptions.Strict))
    values = {
        'rdk_sa_score': sa,
        'rdk_qed': float(QED.qed(mol)),
        'rdk_mw': mw,
        'rdk_clogp': clogp,
        'rdk_tpsa': tpsa,
        'rdk_hbd': hbd,
        'rdk_hba': hba,
        'rdk_rotatable_bonds': rotatable,
        'rdk_formal_charge': int(Chem.GetFormalCharge(mol)),
        'rdk_heavy_atoms': int(mol.GetNumHeavyAtoms()),
        'rdk_aromatic_rings': int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        'rdk_fraction_csp3': float(rdMolDescriptors.CalcFractionCSP3(mol)),
        'rdk_ro5_violations': sum((mw > 500, clogp > 5, hbd > 5, hba > 10)),
        'rdk_veber_pass': int(rotatable <= 10 and tpsa <= 140),
        # Counts of matching catalog entries, not occurrences/atoms or booleans.
        'rdk_pains_alerts': len(_catalog('PAINS').GetMatches(mol)),
        'rdk_brenk_alerts': len(_catalog('BRENK').GetMatches(mol)),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(f'RDKit returned nonfinite properties for {smiles!r}')
    return values


def _calculate_input(values):
    return calculate_rdkit_descriptors(*values)


def calculate_rdkit_descriptors_batch(compounds, cpus=1):
    """Calculate ID/SMILES/existing-SA rows locally without subprocess DB access."""
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise ValueError('sa-score-cpus must be a positive integer')
    compounds = list(compounds)
    if not compounds:
        return []
    available = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                 else (os.cpu_count() or 1))
    workers = min(cpus, available, len(compounds))
    ids = [row[0] for row in compounds]
    inputs = [(row[1], row[2]) for row in compounds]
    if workers == 1:
        values = [_calculate_input(value) for value in inputs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunksize = max(1, len(compounds) // (workers * 4))
            values = list(pool.map(_calculate_input, inputs, chunksize=chunksize))
    return list(zip(ids, values))
