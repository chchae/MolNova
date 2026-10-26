"""Pure remote force-field/search helper; never opens SQLite or claims work."""
import json
import sys
from rdkit import Chem
from molnova.chemistry.combichem3d import history as provenance
from molnova.chemistry.combichem3d.search import search
from molnova.chemistry.combichem3d.clashes import BumpChecker, load_receptor
from molnova.chemistry.combichem3d.output import versions


def main():
    cfg = json.loads(sys.stdin.readline())
    if versions() != cfg['runtime-versions']:
        raise RuntimeError('Remote calculator Python/RDKit/NumPy versions differ from execution host')
    checker = BumpChecker(load_receptor(cfg['receptor'], cfg), cfg)
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        product = Chem.MolFromMolBlock(request['molblock'], removeHs=False)
        if product is None:
            raise ValueError('Invalid calculator molecule')
        product = provenance.restore(product, request['history'])
        result = search(product, request['history'], checker, cfg, request['seed'], deadline=request['deadline'])
        body = {key: value for key, value in result.items() if key != 'pose'}
        if result.get('pose') is not None:
            body['molblock'] = Chem.MolToMolBlock(result['pose'])
            body['coordinates'] = result['pose'].GetConformer().GetPositions().tolist()
        print(json.dumps(body), flush=True)


if __name__ == '__main__':
    main()
