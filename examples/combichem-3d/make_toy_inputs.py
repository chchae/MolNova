"""Create an isolated artificial geometry example, never a production project.

python make_toy_inputs.py /tmp/molnova-3d-demo --start-iteration 1
"""
import argparse
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem


def create(directory, start_iteration=1, attachments=1):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Toy destination must be empty')
    parent = Chem.RWMol(Chem.MolFromSmiles('c1ccccc1'))
    atoms = [dict(index=i,uid=f'core:{i}',branch='core',fragment='core',added_iteration=0) for i in range(6)]
    if start_iteration == 1:
        for branch in range(attachments):
            a = parent.AddAtom(Chem.Atom(6)); b = parent.AddAtom(Chem.Atom(6))
            parent.AddBond(branch,a,Chem.BondType.SINGLE); parent.AddBond(a,b,Chem.BondType.SINGLE)
            atoms.extend(dict(index=index,uid=f'branch{branch+1}:{j}',branch=f'branch_{branch+1}',
                              fragment=f'iter1_fragment_{branch+1}',added_iteration=1) for j,index in enumerate((a,b)))
    parent = parent.GetMol()
    Chem.SanitizeMol(parent)
    parent = Chem.AddHs(parent)
    if AllChem.EmbedMolecule(parent,randomSeed=37) < 0:
        raise RuntimeError('Toy parent embedding failed')
    AllChem.UFFOptimizeMolecule(parent,maxIters=1000)
    parent = Chem.RemoveHs(parent)
    parent.SetProp('_Name','artificial_parent')
    with Chem.SDWriter(str(directory/'parents.sdf')) as writer:
        writer.write(parent)
    history = dict(schema='molnova_pose_history_v1',iteration=start_iteration,
                   core_smarts='c1ccccc1',atoms=atoms,attachments=[f'core:{i}' for i in range(attachments)],
                   core=[dict(core_index=i,uid=f'core:{i}',xyz=list(parent.GetConformer().GetAtomPosition(i))) for i in range(6)])
    (directory/'history.json').write_text(json.dumps(dict(parents=[history]),indent=2)+'\n')
    receptor = Chem.MolFromSmiles('[Ne]')
    conf = Chem.Conformer(1); conf.Set3D(True); conf.SetAtomPosition(0,(30.,30.,30.)); receptor.AddConformer(conf)
    receptor.SetProp('_Name','artificial_distant_receptor_atom')
    with Chem.SDWriter(str(directory/'receptor.sdf')) as writer:
        writer.write(receptor)
    (directory/'sidechains.smi').write_text('[*]C methyl\n[*:9]F fluorine\n[*]CC ethyl\n-H keep_hydrogen\n')
    references = ['c1ccccc1','Cc1ccccc1','CCc1ccccc1','CCCc1ccccc1','Fc1ccccc1','Clc1ccccc1','Brc1ccccc1',
                  'Oc1ccccc1','Nc1ccccc1','COc1ccccc1','CCOc1ccccc1','O=C(O)c1ccccc1','O=C(N)c1ccccc1',
                  'CC(=O)c1ccccc1','c1ccncc1','CCOc1ccncc1','CNc1ccccc1','CCNc1ccccc1',
                  'CC(C)c1ccccc1','CC(C)Oc1ccccc1']
    (directory/'references.smi').write_text('\n'.join(references)+'\n')
    settings = f'''# Artificial geometry smoke example; not a binding-site model.
# For a real existing input TOML, append project-settings.example.toml tables;
# reuse its common inputs. The 30-second/two-process values below are toy values.
receptor = "receptor.sdf"
parents = "parents.sdf"
history = "history.json"
sidechain-library = "sidechains.smi"
reference-smiles = "references.smi"
out-dir = "output"
growth-mode = "frontier" # legacy artificial frontier regression example
max-iteration = {start_iteration+1}
sidechains-per-site = 3 # small artificial library; retain legacy ordered controls here
sidechain-min-heavy-atoms = 1
sidechain-max-heavy-atoms = 1000
sidechain-selection = "ordered"
sidechain-include-h = true
conformers = 3
torsion-trials = 2
optimization-iterations = 1000
processes = 2
seed = 37
batch-size = 2
max-seconds = 30.0
stop-mode = "time"
target-count = 10000
core-displacement = 0.05
overlap-tolerance = 0.4
one-four-policy = "exclude"
[filters]
mw = [0.0, 2000.0]
qed = [0.0, 1.0]
similarity = [0.0, 1.0]
'''
    (directory/'grow.toml').write_text(settings)
    return directory/'grow.toml'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--start-iteration',type=int,choices=(0,1),default=1)
    parser.add_argument('--attachments',type=int,choices=(1,2,3,4),default=1)
    args = parser.parse_args()
    print(create(args.directory,args.start_iteration,args.attachments))
