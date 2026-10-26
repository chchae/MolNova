"""Seeded bounded candidate pools and full mixed-radix streaming, with optional -H."""
import math
import hashlib
import random
from rdkit import Chem, rdBase


def normalize(smiles):
    if smiles in ('-H', 'H', '[H]', '[*][H]', '[*]-[H]'):
        return '-H'
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('Fragment must be one valid connected SMILES')
    dummy = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if (len(dummy) != 1 or dummy[0].GetDegree() != 1
            or dummy[0].GetBonds()[0].GetBondType() != Chem.BondType.SINGLE):
        raise ValueError('Fragment must have exactly one terminal single-bond [*] or [*:n]')
    if dummy[0].GetNeighbors()[0].GetAtomicNum() == 1:
        return '-H'
    if dummy[0].GetNeighbors()[0].GetAtomicNum() <= 1:
        raise ValueError('Fragment attachment must connect to a heavy atom or -H')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_library(path, cfg=None):
    cfg=cfg or {}
    accepted, rejected, seen = (['-H'] if cfg.get('sidechain-include-h',True) else []), [], {'-H'}
    with open(path) as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            token = line.split()[0]
            try:
                value = normalize(token)
                if value not in seen:
                    heavy=Chem.MolFromSmiles(value).GetNumHeavyAtoms()
                    minimum,maximum=cfg.get('sidechain-min-heavy-atoms',1),cfg.get('sidechain-max-heavy-atoms',1000000)
                    if not minimum<=heavy<=maximum:
                        rejected.append(dict(line=line_no,smiles=token,heavy_atoms=heavy,
                                             reason=f'Sidechain heavy atom count outside [{minimum}, {maximum}]; dummy and H excluded'))
                        seen.add(value)
                        continue
                    accepted.append(value)
                    seen.add(value)
            except (ValueError, RuntimeError) as exc:
                rejected.append(dict(line=line_no, smiles=token, reason=str(exc)))
    if not any(value!='-H' for value in accepted):
        raise ValueError('Sidechain library has no valid heavy-atom fragments within the configured size range')
    return accepted, rejected


def candidate_pools(sites, library, cfg):
    k = cfg['sidechains-per-site']
    if k > len(library):
        raise ValueError(f'sidechains-per-site={k} exceeds {len(library)} eligible unique candidates; reduce K or broaden the size range')
    available = set(library)
    pools = []
    for site in sites:
        selected = cfg['site-candidates'].get(site)
        if selected is None:
            if cfg.get('sidechain-selection','ordered')=='random':
                seed=int.from_bytes(hashlib.sha256(f'{cfg["seed"]}:{site}'.encode()).digest(),'big')
                include_h='-H' in available
                selected=(['-H'] if include_h else [])+random.Random(seed).sample(sorted(s for s in library if s!='-H'),k-int(include_h))
            else:
                selected=library[:k]
        normalized = list(dict.fromkeys(normalize(s) for s in selected))
        if not normalized or not set(normalized) <= available:
            raise ValueError(f'Site {site}: every candidate must occur in the validated library (or be -H)')
        if len(normalized)>k:
            raise ValueError(f'Site {site}: candidate count exceeds sidechains-per-site={k}')
        pools.append(normalized)
    return pools


def combination_at(pools, ordinal):
    total = math.prod(len(p) for p in pools)
    if not pools or not 0 <= ordinal < total:
        raise IndexError('Cartesian ordinal outside the parent combination space')
    result = []
    for pool in reversed(pools):
        ordinal, digit = divmod(ordinal, len(pool))
        result.append(pool[digit])
    return list(reversed(result))
