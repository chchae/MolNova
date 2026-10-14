"""Synthetic accessibility using RDKit's bundled, unmodified SA scorer."""
import math
import os
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem


def calculate_sa_score(smiles: str) -> float:
    from rdkit.Contrib.SA_Score import sascorer

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Cannot calculate SA score for invalid or empty SMILES: {smiles!r}")
    score = float(sascorer.calculateScore(mol))
    if not math.isfinite(score) or not 1 <= score <= 10:
        raise ValueError(f"RDKit returned an invalid SA score: {score}")
    return score


def calculate_sa_scores(compounds, cpus=1):
    """Score an ID/SMILES batch locally; calculation processes never access SQLite."""
    if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
        raise ValueError('sa-score-cpus must be a positive integer')
    compounds = list(compounds)
    if not compounds:
        return []
    available = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                 else (os.cpu_count() or 1))
    workers = min(cpus, available, len(compounds))
    ids, smiles = zip(*compounds)
    if workers == 1:
        scores = [calculate_sa_score(value) for value in smiles]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            chunksize = max(1, len(compounds) // (workers * 4))
            scores = list(pool.map(calculate_sa_score, smiles, chunksize=chunksize))
    return list(zip(ids, scores))
