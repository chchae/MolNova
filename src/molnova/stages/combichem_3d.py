"""Local SQLite-coordinated combichem-3d workers and sequential supervision."""
import argparse
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import time
import traceback
from rdkit import Chem

from molnova import database
from molnova.config import combichem_3d_settings
from molnova.states import PoseVerdict
from molnova.chemistry.combichem3d import history as provenance
from molnova.chemistry.combichem3d.assembly import assemble, growth_sites
from molnova.chemistry.combichem3d.clashes import BumpChecker, load_receptor
from molnova.chemistry.combichem3d.evaluation import Evaluator
from molnova.chemistry.combichem3d.library import load_library, candidate_pools, combination_at
from molnova.chemistry.combichem3d.output import export_results, versions, log_frames, log_iteration_results
from molnova.chemistry.combichem3d.search import methyl_probe, search, core_deviation
from molnova.chemistry.combichem3d.logging_utils import logger, configure, progress, trace


def stable_seed(seed, *parts):
    digest = hashlib.sha256(json.dumps([seed,*parts],sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:4],'big') % 2147483646


def serial_report(report):
    return {key:value for key,value in report.items() if key != 'pose'}


def restore_parent(row):
    mol = Chem.MolFromMolBlock(row['molblock'],removeHs=False)
    if mol is None:
        raise ValueError('Cannot parse durable parent pose')
    return provenance.restore(mol,row['history'])


def plan_parent(db, job, cfg, library, checker, deadline):
    row = database.growth_parent(db,job['parent_pose'])
    parent, history = restore_parent(row), row['history']
    sites = growth_sites(parent,history)
    logger.debug('parent_pose=%s planning iteration=%s: eligible_growth_sites=%s',job['parent_pose'],history['iteration']+1,len(sites))
    passed, reports = [], []
    for uid in sites:
        if database.growth_should_stop(db):
            return
        report = row['probes'].get(uid)
        if report is None or report.get('budget_interrupted'):
            trace('probe start parent_pose=%s site=%s',job['parent_pose'],uid)
            result = methyl_probe(parent,history,uid,checker,cfg,
                                  stable_seed(cfg['seed'],history['lineage_id'],uid),deadline)
            report = serial_report(result)
            report['budget_interrupted'] = time.time() >= deadline
            report['gates_enumeration'] = history['iteration'] != 0
            database.growth_save_probe(db,job['parent_pose'],uid,report)
        trace('probe parent_pose=%s site=%s status=%s phase=%s valid_poses=%s attempts=%s reason=%s errors=%s',
                     job['parent_pose'],uid,report['status'],report.get('phase'),report.get('valid_poses'),
                     report.get('attempts'),str(report.get('reason',''))[:400],str(report.get('errors',[]))[:400])
        if report['status'] == PoseVerdict.PASS:
            passed.append(uid)
        reports.append((uid,report))
    if database.growth_should_stop(db):
        return
    # The specified iteration-1 K^N space is not pruned by a methyl proxy:
    # core probes are diagnostic there; every completed product is checked.
    # Later iterations substitute all PASS sites from all newest fragments.
    if history['iteration']==0 or history.get('growth_mode')=='frame-cycles':
        passed = list(history['attachments'])
    pools = candidate_pools(passed,library,cfg) if passed else []
    optional_sites=history.get('growth_mode')=='frame-cycles' and history['iteration']>0
    if optional_sites:
        # Keep the sampled heavy fragments intact; -H is an additional no-op.
        # Existing H-enabled pools already contain it and must not duplicate it.
        pools=[pool if '-H' in pool else ['-H',*pool] for pool in pools]
    plan = dict(sites=passed,pools=pools,total=math.prod(len(pool) for pool in pools) if pools else 0,
                optional_sites=optional_sites,
                eligible_sites=sites,reason=('Full core-attachment Cartesian product' if history['iteration']==0 else
                'Optional PASS-site Cartesian product including -H' if optional_sites else
                'PASS-site Cartesian product') if pools else 'No complete set of eligible PASS growth sites')
    database.growth_save_plan(db,job['parent_pose'],plan,reports)
    logger.debug('parent_pose=%s plan ready: PASS=%s CLASH_REJECT=%s UNRESOLVED=%s selected_sites=%s candidates_per_site=%s combinations=%s',
                job['parent_pose'],sum(r['status']==PoseVerdict.PASS for _,r in reports),
                sum(r['status']==PoseVerdict.CLASH_REJECT for _,r in reports),
                sum(r['status']==PoseVerdict.UNRESOLVED for _,r in reports),len(passed),[len(p) for p in pools],plan['total'])


def evaluate_batch(db, job, cfg, checker, evaluator, deadline, references=None):
    row = database.growth_parent(db,job['parent_pose'])
    parent,history,plan = restore_parent(row),row['history'],row['plan']
    trace('batch start parent_pose=%s iteration=%s cursor=%s stop=%s',job['parent_pose'],history['iteration']+1,job['cursor'],job['stop'])
    from molnova.stages.combichem_buffer import ResultBuffer
    with ResultBuffer(db, job, cfg) as buffer:
        for ordinal in range(int(job['cursor']),int(job['stop'])):
            if database.growth_should_stop(db):
                return
            choices = combination_at(plan['pools'],ordinal)
            trace('product start parent_pose=%s ordinal=%s choices=%s',job['parent_pose'],ordinal,choices)
            payload = None
            try:
                product,birth = assemble(parent,history,plan['sites'],choices,f'c{ordinal}')
                if references is not None:
                    birth=references.constrain(product,birth)
                result = search(product,birth,checker,cfg,stable_seed(cfg['seed'],history['lineage_id'],ordinal),deadline=deadline)
                report = serial_report(result)
                if result['status'] == PoseVerdict.PASS:
                    pose = result['pose']
                    pruned = False
                    try:
                        properties,passed,rejected = evaluator.evaluate(pose)
                        pruned = cfg['prune-intermediates'] and not passed
                    except (ValueError,RuntimeError) as exc:
                        # Preserve an acquired valid 3D pose even if its output
                        # descriptors cannot be evaluated. Unknown is not pruning.
                        properties,passed,rejected = {},False,['evaluation_unresolved']
                        report.update(status=PoseVerdict.UNRESOLVED,geometry_status=PoseVerdict.PASS,
                                      reason=str(exc),failed_stage='property_evaluation')
                    report.update(output_passed=passed,filter_rejections=rejected,properties=properties,
                                  intermediate_pruned=pruned)
                    payload = dict(smiles=provenance.canonical_smiles(pose),molblock=Chem.MolToMolBlock(pose),
                                   history=provenance.snapshot(pose,birth),properties=properties,output_passed=passed,
                                   pruned=pruned)
            except (ValueError,RuntimeError) as exc:
                report = dict(status=PoseVerdict.UNRESOLVED,reason=str(exc),failed_stage='assembly_or_evaluation')
            report.update(iteration=history['iteration']+1,choices=dict(zip(plan['sites'],choices)),
                          expanded_sites=[uid for uid,choice in zip(plan['sites'],choices) if choice!='-H'],
                          expanded_site_count=sum(choice!='-H' for choice in choices))
            buffer.append(ordinal,report,payload)
            trace('product parent_pose=%s ordinal=%s status=%s output_passed=%s valid_poses=%s attempts=%s reason=%s errors=%s filters=%s clashes=%s',
                         job['parent_pose'],ordinal,report['status'],report.get('output_passed',False),report.get('valid_poses'),
                         report.get('attempts'),str(report.get('reason',''))[:400],str(report.get('errors',[]))[:400],
                         report.get('filter_rejections',[]),str(report.get('clashes'))[:600])
    database.growth_finish_batch(db,job)


def worker_loop(db, cfg, iteration):
    # Independent calculators inspect/claim SQLite; they never invoke each other.
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    signal.signal(signal.SIGTERM,signal.SIG_DFL)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    job = None
    try:
        logger.debug('worker initialization iteration=%s: loading receptor, references and sidechain library',iteration)
        checker = BumpChecker(load_receptor(cfg['receptor'],cfg),cfg)
        evaluator = Evaluator(cfg['reference-smiles'],cfg['filters'])
        library,_ = load_library(cfg['sidechain-library'],cfg)
        references=None
        if cfg.get('growth-mode')=='frame-cycles':
            from molnova.chemistry.combichem3d.frames import ReferencePoses
            references=ReferencePoses(cfg['reference-poses'],cfg['reference-mcs'])
        deadline = database.growth_run_info(db)['deadline']
        while True:
            job = database.growth_claim(db,iteration,cfg['batch-size'],os.getpid())
            if job is None:
                logger.debug('worker finished iteration=%s: no claimable work',iteration)
                return
            if job['kind'] == 'wait':
                time.sleep(.05)
                continue
            if job['kind'] == 'input':
                row = database.growth_parent(db,job['parent_pose'])
                parent = restore_parent(row)
                properties,passed,rejected = evaluator.evaluate(parent)
                baseline_pose = Chem.AddHs(parent,addCoords=True)
                baseline = checker.check(baseline_pose)
                baseline.update(kind='baseline',reason='Input pose before growth; reported separately, not silently subtracted',
                                filter_rejections=rejected,generated_hydrogens=baseline_pose.GetNumAtoms()-parent.GetNumAtoms())
                database.growth_save_input(db,job['parent_pose'],properties,passed and baseline['passed'],baseline,
                                           prune=cfg['prune-intermediates'] and not passed,
                                           pose_data=dict(molblock=Chem.MolToMolBlock(baseline_pose),
                                                          history=provenance.snapshot(baseline_pose,row['history'])))
                logger.debug('baseline parent_pose=%s passed=%s property_filters_passed=%s receptor_clashes=%s internal_clashes=%s',
                             job['parent_pose'],baseline['passed'],passed,baseline['receptor_clashes'],baseline['internal_clashes'])
            elif job['kind']=='frame':
                from molnova.chemistry.combichem3d.cycles import prepare_next_frame
                prepare_next_frame(db,iteration,cfg,checker,deadline)
            elif job['kind'] == 'plan':
                plan_parent(db,job,cfg,library,checker,deadline)
            else:
                evaluate_batch(db,job,cfg,checker,evaluator,deadline,references)
    except BaseException:
        logger.exception('worker failed iteration=%s job=%s',iteration,job)
        database.growth_log_event(db,'worker_error',dict(reason=traceback.format_exc(),owner=os.getpid(),job=job))
        if job and 'parent_pose' in job:
            database.growth_save_probe(db,job['parent_pose'],f'__worker_error__:{os.getpid()}',
                dict(status=PoseVerdict.UNRESOLVED,reason=traceback.format_exc(),failed_stage='worker'))
        database.growth_request_stop(db,'worker_error')
        raise


def materialize_project_references(cfg):
    """Reuse reference identities from shared project inputs, with no DB writes."""
    if not cfg.get('reference-source-db') and not cfg.get('reference-source-poses'):
        return
    if cfg.get('reference-source-db'):
        logger.info('loading original reference SMILES from read-only project DB: %s',cfg['reference-source-db'])
        records=database.growth_project_references(cfg['reference-source-db'])
        mols=[(cid,Chem.MolFromSmiles(smiles)) for cid,smiles in records]
    else:
        source=Path(cfg['reference-source-poses'])
        if str(source)==cfg.get('native-parent-source'):
            source=Path(cfg.get('reference-poses',cfg['parents']))
        if source.suffix.lower()!='.sdf' or not source.is_file():
            raise ValueError('No readable project reference DB/reference-pose SDF; add reference-smiles to the input TOML')
        mols=list(enumerate(Chem.SDMolSupplier(str(source),removeHs=False),1))
    if not mols or any(mol is None for _,mol in mols):
        raise ValueError('Project references must contain valid molecules; no reference rows were silently dropped')
    text=''.join(f'{provenance.canonical_smiles(mol)}\tCMPID_{cid}\n' for cid,mol in mols)
    logger.info('project references ready: records=%s cache=%s',len(mols),cfg['reference-smiles'])
    destination=Path(cfg['reference-smiles'])
    if destination.is_file():
        if destination.read_text()!=text:
            raise RuntimeError('Project reference identities changed; cached inputs retained. Use a new 3D output directory.')
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.smi.tmp')
    temporary.write_text(text)
    os.replace(temporary,destination)


def prepare_project(cfg):
    frame_mode=cfg.get('growth-mode')=='frame-cycles'
    if frame_mode and cfg['max-iteration']>1 and not cfg['elite-rank-by']:
        raise ValueError('Set [combichem-3d].elite-rank-by to qed or similarity before multi-cycle elite selection; no affinity score is calculated by this module')
    logger.info('preflight: parents=%s receptor=%s history=%s library=%s',cfg['parents'],cfg['receptor'],cfg['history'],cfg['sidechain-library'])
    cfg['runtime-versions'] = versions()
    output = Path(cfg['out-dir'])
    db = output/'combichem3d.sqlite'
    allowed = {db.name,db.name+'-wal',db.name+'-shm',db.name+'.combichem3d.driver.lock','run.log'}
    from molnova.chemistry.combichem3d.inputs import native_sources, materialize_native_inputs
    if frame_mode or cfg.get('reference-source-db') or cfg.get('reference-source-poses') or native_sources(cfg) or cfg.get('native-input-signature') or cfg.get('history-source')=='inferred':
        allowed.add('input-cache')
    if output.exists() and not db.exists() and any(p.name not in allowed for p in output.iterdir()):
        raise ValueError('Use an empty, dedicated output directory; existing project files will not be reused')
    for key in ('parents','receptor','history','sidechain-library'):
        if key=='history' and (frame_mode or cfg.get('history-source')=='inferred'):
            continue
        if not Path(cfg[key]).is_file():
            extra = '; explicit heavy-atom UID/core/branch/fragment/added_iteration history is required' if key=='history' else ''
            raise ValueError(f'Missing input {key}: {cfg[key]}{extra}')
    materialize_native_inputs(cfg)
    if Path(cfg['parents']).suffix.lower()!='.sdf':
        raise ValueError('combichem-3d needs parent SDF with matching atom-history indices; export the project iteration-1 poses and set [combichem-3d].parents')
    if Path(cfg['receptor']).suffix.lower() not in ('.pdb','.sdf'):
        raise ValueError('combichem-3d needs prepared receptor PDB/SDF; export the existing receptor without changing coordinates and set [combichem-3d].receptor')
    from molnova.chemistry.combichem3d.inference import materialize_inferred_history
    materialize_project_references(cfg)
    if frame_mode:
        from molnova.chemistry.combichem3d.frames import prepare_frame_inputs
        prepare_frame_inputs(cfg)
    else:
        materialize_inferred_history(cfg)
    if not Path(cfg['reference-smiles']).is_file():
        raise ValueError(f'Missing input reference-smiles: {cfg["reference-smiles"]}')
    parents = provenance.load_parents(cfg['parents'],cfg['history'],cfg.get('reference-mcs'))
    logger.info('parents validated: records=%s imported_iteration=%s',len(parents),parents[0][1]['iteration'])
    if parents[0][1]['iteration'] > cfg['max-iteration']:
        raise ValueError('max-iteration is the absolute final iteration and cannot precede input parents')
    logger.info('loading sidechain library: %s',cfg['sidechain-library'])
    library,rejected = load_library(cfg['sidechain-library'],cfg)
    logger.info('sidechain library ready: eligible_candidates=%s rejected=%s K=%s',len(library),len(rejected),cfg['sidechains-per-site'])
    # Validate all supplied candidate overrides and K before launching calculators.
    candidate_pools(list(cfg['site-candidates']) or ['__default__'],library,cfg)
    known_uids = {r['uid'] for _,h in parents for r in h['atoms']}
    if set(cfg['site-candidates']) - known_uids:
        raise ValueError('site-candidates keys must identify explicit imported parent atom UIDs')
    records = []
    for root_index,(mol,history) in enumerate(parents):
        if history['iteration']==0 and set(growth_sites(mol,history)) != set(history['attachments']):
            raise ValueError('Each initial core attachment must have a replaceable H; supply explicit valid attachment UIDs')
        if core_deviation(mol,history) > cfg['core-displacement']:
            raise ValueError('Input parent core coordinates exceed core-displacement; confirm the explicit coordinate frame/map')
        history['lineage_id'] = hashlib.sha256(json.dumps([root_index,history],sort_keys=True).encode()).hexdigest()
        records.append(dict(smiles=provenance.canonical_smiles(mol),molblock=Chem.MolToMolBlock(mol),history=history))
    operational = {'out-dir','config-path','max-seconds','target-count','processes','batch-size','grace-seconds','max-iteration','stop-mode','log-level','log-interval','log-structures','execution-host','remote-env-script','result-buffer-size','result-buffer-seconds'}
    frozen = {k:v for k,v in cfg.items() if k not in operational}
    frozen['input_sha256'] = {k:hashlib.sha256(Path(cfg[k]).read_bytes()).hexdigest()
                              for k in ('parents','receptor','history','reference-smiles','sidechain-library')}
    if frame_mode:
        frozen['input_sha256']['reference-poses']=hashlib.sha256(Path(cfg['reference-poses']).read_bytes()).hexdigest()
        frozen['growth_search_policy']=dict(optional_sites_after_initial=True,
                                           movable_atoms='all_sidechains_outside_original_iteration0_core')
    frozen['result_buffer_implementation_sha256']=hashlib.sha256(Path(__file__).with_name('combichem_buffer.py').read_bytes()).hexdigest()
    scientific_source = Path(__file__).resolve().parents[1]/'chemistry/combichem3d'
    frozen['implementation_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(scientific_source.glob('*.py'))}
    frozen['implementation_sha256']['stages/combichem_3d.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    frozen['library_candidates'] = len(library)
    frozen['library_selection'] = dict(method=cfg['sidechain-selection'],seed=cfg['seed'],
                                       minimum_heavy_atoms=cfg['sidechain-min-heavy-atoms'],
                                       maximum_heavy_atoms=cfg['sidechain-max-heavy-atoms'],
                                       include_h=cfg['sidechain-include-h'],candidates_per_site=cfg['sidechains-per-site'])
    fingerprint = hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    try:
        created=database.growth_initialize(db,cfg,fingerprint,records)
    except RuntimeError as exc:
        if not str(exc).startswith('Frozen pose-growth inputs or scientific settings changed'):
            raise
        changed=[]
        manifest=output/'input_manifest.json'
        if manifest.is_file():
            saved=json.loads(manifest.read_text())
            for key in sorted(set(saved)|set(frozen)):
                if saved.get(key)==frozen.get(key):
                    continue
                if key in ('implementation_sha256','input_sha256'):
                    before=saved.get(key,{})
                    after=frozen.get(key,{})
                    changed.extend(f'{key}.{name}' for name in sorted(set(before)|set(after))
                                   if before.get(name)!=after.get(name))
                else:
                    changed.append(key)
        detail=', '.join(changed) or 'saved fingerprint (detailed manifest unavailable)'
        raise RuntimeError(
            f'Frozen pose-growth inputs, scientific settings or implementation changed: {detail}. '
            f'Existing results retained at {output}. Set a different [combichem-3d].out-dir '
            'in the TOML for a fresh calculation; do not delete the existing output. '
            'Use --export-only to export the saved results without recalculation. '
            'Resume in the same output requires the original inputs, scientific settings and implementation.'
        ) from exc
    logger.info('database %s: %s', 'created' if created else 'resume validated',db)
    (output/'input_manifest.json').write_text(json.dumps(frozen,indent=2,sort_keys=True)+'\n')
    with (output/'library_rejected.jsonl').open('w') as stream:
        for row in rejected:
            stream.write(json.dumps(row)+'\n')
    return db,parents[0][1]['iteration']


def supervise(cfg, db, first_iteration):
    """Only lifecycle supervision here: every scientific calculation is a worker."""
    # Recovery does no scientific calculation. An independent stage process
    # persists orphaned journals before any calculator can reclaim their jobs.
    if any((Path(db).parent/'result-buffer').glob('batch-*.jsonl')):
        from molnova.stages.combichem_buffer import recover
        recovery = multiprocessing.get_context('fork').Process(target=recover, args=(db,))
        recovery.start()
        recovery.join()
        if recovery.exitcode != 0:
            raise RuntimeError('Result-buffer recovery failed; journals and claims retained')
    database.growth_start_session(db,cfg)
    processes = []
    stopped = False
    previous = {}
    def interrupt(signum, frame):
        nonlocal stopped
        stopped = True
    for signum in (signal.SIGINT,signal.SIGTERM):
        previous[signum] = signal.signal(signum,interrupt)
    try:
        # fork inherits the exclusive stage lock. If the supervisor is killed,
        # a restart cannot rearm claims until its calculators have exited.
        context = multiprocessing.get_context('fork')
        for iteration in range(first_iteration,cfg['max-iteration']+1):
            if stopped:
                database.growth_request_stop(db,'interrupted')
            if database.growth_should_stop(db):
                break
            logger.info('iteration %s: starting %s local workers',iteration,cfg['processes'])
            source_iteration=first_iteration if iteration==first_iteration else iteration-1
            log_frames(db,source_iteration,iteration,cfg['log-structures'],cfg.get('growth-mode')=='frame-cycles')
            processes = []
            for _ in range(cfg['processes']):
                process = context.Process(target=worker_loop,args=(db,cfg,iteration))
                process.start()
                processes.append(process)
            stop_at = None
            next_progress=0.
            while any(process.is_alive() for process in processes):
                if stopped:
                    database.growth_request_stop(db,'interrupted')
                reason = database.growth_should_stop(db)
                if any(process.exitcode not in (None,0) for process in processes):
                    database.growth_request_stop(db,'worker_error')
                    reason = 'worker_error'
                if reason:
                    if stop_at is None:
                        logger.info('stop requested: reason=%s grace=%.1fs; preserving committed results',reason,cfg['grace-seconds'])
                    stop_at = stop_at or time.monotonic()
                    if time.monotonic()-stop_at >= cfg['grace-seconds']:
                        for process in processes:
                            if process.is_alive():
                                process.terminate()
                if time.monotonic()>=next_progress:
                    progress(database.growth_run_info(db),iteration,sum(p.is_alive() for p in processes))
                    next_progress=time.monotonic()+cfg['log-interval']
                for process in processes:
                    process.join(timeout=.02)
            for process in processes:
                process.join()
            logger.info('iteration %s workers finished: exit_codes=%s',iteration,[p.exitcode for p in processes])
            if any(process.exitcode != 0 for process in processes):
                database.growth_request_stop(db,'worker_error')
            log_iteration_results(db,cfg,iteration,source_iteration)
            if stopped or database.growth_should_stop(db):
                break
        database.growth_end_session(db,'exhausted')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        for signum,handler in previous.items():
            signal.signal(signum,handler)
        database.growth_end_session(db,'interrupted' if stopped else 'worker_error')


def _main(argv=None):
    invocation_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project_toml',type=Path)
    parser.add_argument('--export-only',action='store_true',help='Rebuild artifacts from the dedicated DB without scientific work')
    parser.add_argument('--debug',action='store_true',help='Compact parent and worker diagnostics')
    parser.add_argument('--trace',action='store_true',help='Verbose per-probe/product diagnostics')
    ns = parser.parse_args(argv)
    cfg = combichem_3d_settings(ns.project_toml)
    from molnova.stages.combichem_remote import dispatch
    if dispatch(cfg, ns):
        return None
    if ns.debug:
        cfg['log-level']='DEBUG'
    if ns.trace:
        cfg['log-level']='TRACE'
    db = Path(cfg['out-dir'])/'combichem3d.sqlite'
    with database.stage_lock(db,'combichem3d'):
        configure(cfg['out-dir'],cfg['log-level'])
        logger.info('starting config=%s seed=%s processes=%s max_iteration=%s K=%s conformers=%s torsion_trials=%s time_limit=%.1fs target=%s mode=%s log=%s',
                    cfg['config-path'],cfg['seed'],cfg['processes'],cfg['max-iteration'],cfg['sidechains-per-site'],
                    cfg['conformers'],cfg['torsion-trials'],cfg['max-seconds'],cfg['target-count'],cfg['stop-mode'],Path(cfg['out-dir'])/'run.log')
        if not ns.export_only:
            db,first_iteration = prepare_project(cfg)
            supervise(cfg,db,first_iteration)
        elif not db.is_file():
            raise ValueError('No combichem-3d DB to export')
        logger.info('exporting committed results: %s',cfg['out-dir'])
        info = export_results(db,cfg['out-dir'],cfg['batch-size'])
        info['invocation_wall_seconds'] = time.monotonic()-invocation_started
        temporary = Path(cfg['out-dir'])/'.summary.json.tmp'
        temporary.write_text(json.dumps(info,indent=2,sort_keys=True)+'\n')
        os.replace(temporary,Path(cfg['out-dir'])/'summary.json')
        logger.info('%s: attempted=%s unique=%s elapsed=%.2fs partial=%s; exports ready',
                    info['stop_reason'],info['attempted'],info['unique_count'],info['elapsed'],info['partial_search'])
        from molnova.chemistry.combichem3d.analysis import log_final_analysis
        log_final_analysis(db)
        if info['stop_reason']=='worker_error':
            raise RuntimeError('A local worker failed; results and failure records were preserved')
    return info


def main(argv=None):
    try:
        return _main(argv)
    except Exception:
        if logger.handlers:
            logger.exception('run failed; existing inputs and committed results retained')
        raise


if __name__ == '__main__':
    main()
