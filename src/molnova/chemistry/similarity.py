from rdkit import Chem, DataStructs
from molnova import _core


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    a = Chem.MolFromSmiles(smiles_a)
    b = Chem.MolFromSmiles(smiles_b)
    if a is None or b is None:
        raise ValueError("Invalid SMILES")
    fa = _core.FP_GENERATOR.GetFingerprint(a)
    fb = _core.FP_GENERATOR.GetFingerprint(b)
    return float(DataStructs.TanimotoSimilarity(fa, fb))
