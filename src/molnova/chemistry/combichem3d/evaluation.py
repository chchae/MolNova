"""Output-only property filters, separate from optional intermediate pruning."""
import math
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator, rdMolDescriptors


class Evaluator:
    def __init__(self, reference_path, filters):
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.references = []
        self.filters = filters
        with open(reference_path) as stream:
            for index, line in enumerate(stream, 1):
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                mol = Chem.MolFromSmiles(line.split()[0])
                if mol is None:
                    raise ValueError(f'Invalid reference SMILES at line {index}')
                self.references.append(self.generator.GetFingerprint(mol))
        if not self.references:
            raise ValueError('At least one valid property/similarity reference SMILES is required')

    def evaluate(self, mol):
        mol = Chem.RemoveHs(Chem.Mol(mol))
        fp = self.generator.GetFingerprint(mol)
        values = dict(mw=float(Descriptors.MolWt(mol)), clogp=float(Crippen.MolLogP(mol)),
                      tpsa=float(rdMolDescriptors.CalcTPSA(mol)), qed=float(QED.qed(mol)),
                      hbd=int(Lipinski.NumHDonors(mol)), hba=int(Lipinski.NumHAcceptors(mol)),
                      rotatable_bonds=int(Lipinski.NumRotatableBonds(mol)), heavy_atoms=mol.GetNumHeavyAtoms(),
                      similarity=float(max(DataStructs.BulkTanimotoSimilarity(fp, self.references))))
        if not all(math.isfinite(x) for x in values.values()):
            raise ValueError('Nonfinite molecular descriptor')
        rejected = [key for key, (low, high) in self.filters.items() if not low <= values[key] <= high]
        return values, not rejected, rejected
