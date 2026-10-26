"""Pure remote calculators; no SQLite connections or workflow task claims."""
import json
import logging
import sys
import time
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
    evaluator = references = None
    for line in incoming:
        request=json.loads(line)
        kind = request.get('kind')
        if kind in ('batch', 'properties', 'baseline'):
            if evaluator is None:
                from molnova.chemistry.combichem3d.evaluation import Evaluator
                evaluator = Evaluator(cfg['reference-smiles'],cfg['filters'])
            if kind == 'batch':
                from molnova.stages.combichem_product import evaluate_product
                if references is None and cfg.get('growth-mode')=='frame-cycles':
                    from molnova.chemistry.combichem3d.frames import ReferencePoses
                    references = ReferencePoses(cfg['reference-poses'],cfg['reference-mcs'])
                parent = Chem.MolFromMolBlock(request['molblock'],removeHs=False)
                if parent is None:raise ValueError('Invalid batch parent')
                parent = provenance.restore(parent,request['history'])
                next_ordinal = request['cursor']
                for ordinal in range(request['cursor'],request['stop']):
                    if time.time() >= request['deadline']:break
                    report,payload = evaluate_product(parent,request['history'],request['plan'],ordinal,
                                                     cfg,checker,evaluator,request['deadline'],references)
                    print(json.dumps(dict(_product=ordinal,report=report,payload=payload)),file=outgoing,flush=True)
                    next_ordinal = ordinal+1
                completion = ({'batch_done':True} if next_ordinal==request['stop']
                              else {'batch_interrupted':next_ordinal})
                print(json.dumps(completion),file=outgoing,flush=True)
                continue
            mol = Chem.MolFromMolBlock(request['molblock'],removeHs=False)
            if mol is None:raise ValueError('Invalid evaluation molecule')
            for index,xyz in enumerate(request['coordinates']):
                mol.GetConformer().SetAtomPosition(index,xyz)
            result = evaluator.evaluate(mol) if kind=='properties' else checker.check(mol)
            print(json.dumps({'value':result}),file=outgoing,flush=True)
            continue
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
