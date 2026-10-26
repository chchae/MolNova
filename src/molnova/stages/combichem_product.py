"""One pure product calculation, shared by local workers and Slurm nodes."""
from rdkit import Chem
from molnova.states import PoseVerdict
from molnova.chemistry.combichem3d import history as provenance
from molnova.chemistry.combichem3d.assembly import assemble
from molnova.chemistry.combichem3d.library import combination_at
from molnova.chemistry.combichem3d.search import search


def evaluate_product(parent, history, plan, ordinal, cfg, checker, evaluator,
                     deadline, references=None, *, search_fn=search):
    from molnova.stages.combichem_3d import stable_seed, serial_report
    choices = combination_at(plan['pools'], ordinal)
    payload = None
    try:
        product,birth = assemble(parent,history,plan['sites'],choices,f'c{ordinal}')
        if references is not None:
            birth=references.constrain(product,birth)
        result = search_fn(product,birth,checker,cfg,stable_seed(cfg['seed'],history['lineage_id'],ordinal),deadline=deadline)
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
    return report, payload
