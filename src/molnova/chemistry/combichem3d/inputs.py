"""Immutable, audited exports from reference-pose and Glide grid archives."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from rdkit import Chem
from molnova.chemistry.combichem3d.clashes import SpatialIndex
from molnova.chemistry.combichem3d.native_extract import receptor_member
from molnova.chemistry.combichem3d.logging_utils import logger


def native_sources(cfg):
    result={}
    if Path(cfg['parents']).suffix.lower() in ('.mae','.maegz'):
        result['parents']=cfg['parents']
    if Path(cfg['receptor']).suffix.lower()=='.zip':
        result['grid']=cfg['receptor']
    return result


def digest(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def validate_exports(directory,audit):
    if 'parents' in audit:
        mols=list(Chem.SDMolSupplier(str(directory/'parents.sdf'),removeHs=False))
        if len(mols)!=len(audit['parents']):
            raise ValueError('Native parent export changed the ligand structure count')
        for mol,original in zip(mols,audit['parents']):
            if mol is None or mol.GetNumConformers()!=1:
                raise ValueError('Native parent export produced an invalid ligand')
            if ([a.GetAtomicNum() for a in mol.GetAtoms()]!=original['atomic_numbers']
                    or [a.GetFormalCharge() for a in mol.GetAtoms()]!=original['formal_charges']
                    or mol.GetNumBonds()!=original['bond_count']):
                raise ValueError('Native parent export changed atom order/identity, charge or bond count; history cannot be guessed')
            xyz=np.asarray(mol.GetConformer().GetPositions())
            if not np.isfinite(xyz).all() or np.max(np.linalg.norm(xyz-np.asarray(original['xyz']),axis=1))>.001:
                raise ValueError('Native parent export reordered/moved atoms or changed coordinate frame')
    if 'receptor' in audit:
        mol=Chem.MolFromPDBFile(str(directory/'receptor.pdb'),sanitize=False,removeHs=False,proximityBonding=False)
        original=audit['receptor']
        if mol is None or mol.GetNumAtoms()!=len(original['xyz']) or mol.GetNumConformers()!=1:
            raise ValueError('Grid receptor export lost atoms or coordinate models')
        xyz=np.asarray(mol.GetConformer().GetPositions())
        if not np.isfinite(xyz).all():
            raise ValueError('Nonfinite exported receptor coordinates')
        index=SpatialIndex(xyz)
        consumed=set()
        for number,point,residue in zip(original['atomic_numbers'],original['xyz'],original['residues']):
            point=np.asarray(point)
            candidates=[i for i in index.near(point,.002) if i not in consumed
                        and mol.GetAtomWithIdx(i).GetAtomicNum()==number
                        and np.linalg.norm(xyz[i]-point)<=.002
                        and mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetResidueName().strip()
                        in ({residue,'HIS'} if residue in ('HID','HIE','HIP') else {residue})]
            if not candidates:
                raise ValueError('Grid receptor export changed atom/residue identity or coordinates')
            consumed.add(min(candidates,key=lambda i:np.linalg.norm(xyz[i]-point)))


def materialize_native_inputs(cfg):
    sources=native_sources(cfg)
    if not sources:
        return
    logger.info('native input extraction: sources=%s',sources)
    if 'grid' in sources:
        receptor_member(sources['grid'])  # validate a unique member; never extract ZIP paths to disk
    helper=Path(__file__).with_name('native_extract.py')
    signature=dict(sources={key:dict(path=str(Path(value).resolve()),sha256=digest(value)) for key,value in sources.items()},
                   helper_sha256=digest(helper),suite=cfg['schrodinger'])
    cache=Path(cfg['out-dir'])/'input-cache/native-inputs'
    if cache.exists():
        logger.info('validating native input cache: %s',cache)
        manifest=cache/'manifest.json'
        if not manifest.is_file():
            raise RuntimeError('Incomplete native input cache retained; reconcile it or use a new output directory')
        saved=json.loads(manifest.read_text())
        if saved['source']!=signature:
            raise RuntimeError('Native input sources changed; existing cache retained. Use a new 3D output directory.')
        if any(not (cache/name).is_file() or digest(cache/name)!=value for name,value in saved['files'].items()):
            raise RuntimeError('Native input cache changed; saved data retained for reconciliation')
    else:
        runner=Path(cfg['schrodinger'])/'run'
        if not runner.is_file():
            raise ValueError('Native reference-pose/grid extraction requires a local Schrödinger installation; set [combichem-3d].schrodinger or SCHRODINGER')
        cache.parent.mkdir(parents=True,exist_ok=True)
        temporary=Path(tempfile.mkdtemp(prefix='.native-',dir=cache.parent))
        try:
            logger.info('exporting native structures with %s; timeout=180s',runner)
            command=[str(runner),str(helper),'--output',str(temporary)]
            for key,value in sources.items():
                command.extend(['--'+key,str(value)])
            result=subprocess.run(command,capture_output=True,text=True,timeout=180)
            if result.returncode:
                raise RuntimeError('Native input extraction failed: '+result.stderr[-4000:]+result.stdout[-1000:])
            audit=json.loads((temporary/'atom_source_map.json').read_text())
            validate_exports(temporary,audit)
            logger.info('native exports validated: parents=%s receptor_atoms=%s excluded_receptor_records=%s',
                        len(audit.get('parents',[])),len(audit.get('receptor',{}).get('xyz',[])),len(audit.get('excluded_receptor_records',[])))
            if any(digest(value)!=signature['sources'][key]['sha256'] for key,value in sources.items()):
                raise RuntimeError('Native input changed during extraction; no cache was published')
            files={p.name:digest(p) for p in temporary.iterdir()}
            (temporary/'manifest.json').write_text(json.dumps(dict(source=signature,files=files),indent=2,sort_keys=True)+'\n')
            os.replace(temporary,cache)
            logger.info('native input cache saved: %s',cache)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    cfg['native-input-signature']=signature
    if 'parents' in sources:
        cfg['native-parent-source']=sources['parents']
        cfg['parents']=str(cache/'parents.sdf')
    if 'grid' in sources:
        cfg['native-grid-source']=sources['grid']
        cfg['receptor']=str(cache/'receptor.pdb')
