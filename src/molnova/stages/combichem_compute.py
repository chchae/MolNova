"""Pure remote calculators; no SQLite connections or workflow task claims."""
import json
import logging
import sys
from rdkit import Chem
from molnova.chemistry.combichem3d import history as provenance
from molnova.chemistry.combichem3d.search import search, methyl_probe
from molnova.chemistry.combichem3d.clashes import BumpChecker, load_receptor
from molnova.chemistry.combichem3d.output import versions
from molnova.chemistry.combichem3d.logging_utils import logger


def mcs(request, cfg, outgoing):
    from molnova.chemistry.combichem3d.cycles import select_mcs
    def emit(value):print(json.dumps(value),file=outgoing,flush=True)
    class Relay(logging.Handler):
        def emit(self, record):emit({'_log':record.getMessage(),'level':record.levelno})
    relay=Relay();level=logger.level;logger.setLevel(logging.INFO);logger.addHandler(relay)
    try:
        selected,report=select_mcs(request['elites'],request['previous'],cfg,request['deadline'],
                                   lambda value:emit({'_checkpoint':value}))
    finally:
        logger.removeHandler(relay);logger.setLevel(level)
    if selected is not None:
        pose,history=selected
        selected=dict(molblock=Chem.MolToMolBlock(pose),history=provenance.snapshot(pose,history))
    return dict(selected=selected,report=report)


def serve(incoming, outgoing):
    cfg=json.loads(incoming.readline())
    if versions()!=cfg['runtime-versions']:
        raise RuntimeError('Remote calculator Python/RDKit/NumPy versions differ from coordinator')
    checker=BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
    print(json.dumps({'ready':True}),file=outgoing,flush=True)
    for line in incoming:
        request=json.loads(line)
        if request.get('kind')=='mcs':
            print(json.dumps(mcs(request,cfg,outgoing)),file=outgoing,flush=True)
            continue
        product=Chem.MolFromMolBlock(request['molblock'],removeHs=False)
        if product is None:raise ValueError('Invalid calculator molecule')
        product=provenance.restore(product,request['history'])
        probe=request.get('kind')=='methyl_probe'
        if probe:
            result=methyl_probe(product,request['history'],request['uid'],checker,cfg,request['seed'],request['deadline'])
        else:
            result=search(product,request['history'],checker,cfg,request['seed'],deadline=request['deadline'])
        body={key:value for key,value in result.items() if key!='pose'}
        if not probe and result.get('pose') is not None:
            body['molblock']=Chem.MolToMolBlock(result['pose'])
            body['coordinates']=result['pose'].GetConformer().GetPositions().tolist()
        print(json.dumps(body),file=outgoing,flush=True)


if __name__=='__main__':serve(sys.stdin,sys.stdout)
