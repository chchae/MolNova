"""Synthetic accessibility using RDKit's bundled, unmodified SA scorer."""
import math

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
