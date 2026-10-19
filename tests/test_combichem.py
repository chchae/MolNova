import csv
import json
import itertools
import math
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from rdkit import Chem

from molnova import _core, cli, database, driver
from molnova.chemistry import combichem
from molnova.chemistry.descriptors import calculate_rdkit_descriptors
from molnova.chemistry.scaffold import prepare_iteration_scaffold
from molnova.stages import generate


def test_library_attachment_validation_deduplication_and_audit(tmp_path):
    library = tmp_path / 'edges.smi'
    library.write_text('[*]N\n[*]Cl\n[*]C first\n*C duplicate\n[*]OC\n'
                       '*CC*\n*=C\nnot_a_smiles\n*C.[Na+]\n'
                       '[*]O[Si](C)(C)C\n[*][S+](N(C)C)[O-]\n')
    report = tmp_path / 'rejected.tsv'
    accepted, fingerprint, rejected = combichem.load_library(library, report)
    assert len(accepted) == 6
    assert rejected == 5
    assert len(fingerprint) == 64
    assert len({row['smiles'] for row in accepted}) == 6
    assert {row['line'] for row in accepted} >= {1, 2}
    audit = list(csv.DictReader(report.open(), delimiter='\t'))
    assert any(row['reason'] == 'duplicate sidechain' for row in audit)
    assert any(row['line'] == '6' for row in audit)


def test_carbon_free_sidechains_are_accepted_and_assembled(tmp_path):
    path = tmp_path / 'edges.smi'
    path.write_text('*N\n*O\n*Cl\n')
    accepted, _, rejected = combichem.load_library(path, tmp_path / 'rejected.tsv')
    assert len(accepted) == 3
    assert rejected == 0
    core = Chem.MolFromSmiles('C[*:0]')
    products = {combichem.assemble(core, [Chem.MolFromSmiles(row['smiles'])])
                for row in accepted}
    assert products == {'CN', 'CO', 'CCl'}


def test_no_eligible_sidechains_is_an_error(tmp_path):
    path = tmp_path / 'edges.smi'
    path.write_text('*\n*CC*\n*=C\n')
    with pytest.raises(RuntimeError, match='No eligible sidechains'):
        combichem.load_library(path, tmp_path / 'rejected.tsv')


def test_weighted_sampling_prefers_small_fragments_without_replacement():
    library = [dict(line=1, smiles='*C', heavy_atoms=1),
               dict(line=2, smiles='*' + 'C' * 20, heavy_atoms=20)]
    small = sum(combichem.select_sidechains(library, random.Random(seed), 1)[0]['line'] == 1
                for seed in range(500))
    assert small > 450
    all_rows = [dict(line=i, smiles=str(i), heavy_atoms=i + 1) for i in range(1200)]
    selected = combichem.select_sidechains(all_rows, random.Random(123))
    assert len(selected) == len({row['line'] for row in selected}) == 1000
    assert selected == combichem.select_sidechains(all_rows, random.Random(123))
    assert selected != combichem.select_sidechains(all_rows, random.Random(124))


@pytest.mark.parametrize('supplied', [None, 'c1ccccc1', '*c1cc(*)cc(*)c1'])
def test_combichem_uses_all_three_attachment_positions(supplied):
    refs = [dict(id=i, name=str(i), mol=Chem.MolFromSmiles(smiles))
            for i, smiles in enumerate(['Clc1cc(Br)cc(I)c1', 'Fc1cc(N)cc(O)c1'])]
    info = combichem.prepare_scaffold(refs, supplied)
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
    assert len(combichem.attachment_atoms(scaffold)) == 3
    product = combichem.assemble(scaffold, [Chem.MolFromSmiles('*CC')] * 3)
    assert product == Chem.MolToSmiles(Chem.MolFromSmiles('CCc1cc(CC)cc(CC)c1'))
    assert '*' not in product


def test_automatic_combichem_accepts_one_attachment_instead_of_libinvent_fallback():
    refs = [dict(id=i, name=str(i), mol=Chem.MolFromSmiles(smiles))
            for i, smiles in enumerate(['COc1ccc(Cl)cc1', 'COc1ccc(F)cc1'])]
    info = combichem.prepare_scaffold(refs, None)
    assert len(combichem.attachment_atoms(Chem.MolFromSmiles(info['scaffold_smiles']))) == 1


def test_all_inferred_external_bonds_on_the_same_core_atom_are_attached():
    refs = [dict(id=i, name=str(i), mol=Chem.MolFromSmiles(smiles))
            for i, smiles in enumerate(['CNC', 'CCNCC'])]
    info = combichem.prepare_scaffold(refs, 'N')
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
    assert len(combichem.attachment_atoms(scaffold)) == 2
    assert combichem.assemble(scaffold, [Chem.MolFromSmiles('*C')] * 2) == 'CNC'


def test_inferred_multiple_bond_attachment_is_rejected():
    refs = [dict(id=i, name=str(i), mol=Chem.MolFromSmiles(smiles))
            for i, smiles in enumerate(['C=C', 'C=O'])]
    with pytest.raises(RuntimeError, match='not a single bond'):
        combichem.prepare_scaffold(refs, 'C')


@pytest.mark.parametrize('core, sidechain, expected', [
    ('N[C@H](F)[*:0]', '*C', 'N[C@H](F)C'),
    ('[*:0]C', '*C/C=C/Cl', 'CC/C=C/Cl'),
    ('[*:0]C', '*[S+](N(C)C)[O-]', 'C[S+](N(C)C)[O-]'),
    ('[*:0]C', '*O[Si](C)(C)C', 'CO[Si](C)(C)C'),
])
def test_assembly_preserves_stereochemistry_charges_and_elements(core, sidechain, expected):
    actual = combichem.assemble(Chem.MolFromSmiles(core), [Chem.MolFromSmiles(sidechain)])
    assert actual == Chem.MolToSmiles(Chem.MolFromSmiles(expected), isomericSmiles=True)


def test_cartesian_permutation_visits_every_combination_once():
    pools = [[dict(line=i, smiles=str(i)) for i in range(length)] for length in [2, 3, 4]]
    plan = dict(pools=pools, total_combinations='24', offset='7', stride='5')
    found = [tuple(row['line'] for row in combichem.combination_at(plan, i)) for i in range(24)]
    assert len(set(found)) == 24
    assert set(found) == set(itertools.product(range(2), range(3), range(4)))
    with pytest.raises(IndexError):
        combichem.combination_at(plan, 24)


@pytest.fixture
def project(tmp_path, monkeypatch):
    path = tmp_path / 'test.sqlite'
    library = tmp_path / 'edges.smi'
    library.write_text('*C\n*CC\n*OC\n*N(C)C\n')
    with database.connect(path) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany("INSERT INTO compound(name,smiles,iteration,state) VALUES (?,?,0,'reference')",
                         [('R1', 'Clc1ccc(F)cc1'), ('R2', 'Brc1ccc(I)cc1')])
    args = SimpleNamespace(db_path=path, output=tmp_path, sidechain_library=library,
                           generation_method='combichem', sample_size=4, target_count=4,
                           max_iteration=5, gbsa_input_count=2, gbsa_elite_count=2,
                           iteration_mcs=True, initial_mcs_smarts='c1ccccc1',
                           elite_mcs_smarts='[*:0]c1ccc([*:1])cc1', elite_mcs_fixed=False,
                           mcs_timeout=60, sa_score_cpus=1, sa_accepted_ratio=.75,
                           synthetic_feasibility_enabled=False)
    def unexpected(*a, **k):
        pytest.fail('Combichem invoked REINVENT or LibInvent training')
    for name in ['run_libinvent_sampling', 'run_libinvent_elite_transfer_learning',
                 'write_elite_libinvent_training_files']:
        monkeypatch.setattr(_core, name, unexpected)
    return args


@pytest.fixture
def adaptive_project(project, monkeypatch):
    prepare_iteration_scaffold(project, 1, create=True)
    project.elite_mcs_smarts = None
    project.gbsa_elite_count = 1
    project.sample_size = 1
    project.target_count = 100
    monkeypatch.setattr(combichem.secrets, 'choice', lambda candidates: candidates[-1])
    prepare_iteration_scaffold(project, 2, [(100, 'elite', 'CCCc1ccc(CCC)cc1', 1, -10, -40)], create=True)
    combichem.prepare_plan(project, 2, project.output / 'iter2/combichem')
    return project


def test_empty_batch_streak_is_atomic_persistent_and_resets_on_insert(project):
    database.combichem_plan(project.db_path, 1, dict(total_combinations='100'))
    for i in range(9):
        database.insert_generated_candidates(project.db_path, 1, 100, [], combichem_progress=(i, i + 1, 0))
    assert database.combichem_empty_batches(project.db_path, 1) == 9
    # Replayed commits and failed calculations must not count twice.
    database.insert_generated_candidates(project.db_path, 1, 100, [], combichem_progress=(8, 9, 0))
    bad = dict(calculate_rdkit_descriptors('CCO'), smiles='CCO', druglike_selected=True, rdk_qed=float('nan'))
    with pytest.raises(ValueError):
        database.insert_generated_candidates(project.db_path, 1, 100, [bad], combichem_progress=(9, 10, 0))
    assert database.combichem_empty_batches(project.db_path, 1) == 9
    good = dict(calculate_rdkit_descriptors('CCO'), smiles='CCO', druglike_selected=True)
    database.insert_generated_candidates(project.db_path, 1, 100, [good], combichem_progress=(9, 10, 0))
    assert database.combichem_empty_batches(project.db_path, 1) == 0
    for i in range(10, 20):
        database.insert_generated_candidates(project.db_path, 1, 100, [], combichem_progress=(i, i + 1, 0))
    assert database.combichem_empty_batches(project.db_path, 1) == 10
    assert database.combichem_empty_batches(project.db_path, 1, 1) == 0


def test_empty_streak_continues_across_pool_refills(project, monkeypatch):
    prepare_iteration_scaffold(project, 1, create=True)
    monkeypatch.setattr(combichem, 'select_sidechains', lambda library, rng: [dict(row) for row in library[:2]])
    directory = project.output / 'iter1/combichem'
    plan = combichem.prepare_plan(project, 1, directory)
    for round_id in range(2):
        for i in range(4):
            database.insert_generated_candidates(project.db_path, 1, 100, [], combichem_progress=(i, i + 1, round_id))
        plan = database.combichem_plan(project.db_path, 1)
        combichem.refill_plan(project, 1, directory, plan)
    assert database.combichem_empty_batches(project.db_path, 1) == 8
    for i in range(2):
        database.insert_generated_candidates(project.db_path, 1, 100, [], combichem_progress=(i, i + 1, 2))
    assert database.combichem_empty_batches(project.db_path, 1) == 10


def test_supplied_core_does_not_switch_attachments_after_empty_batches(project, monkeypatch):
    project.sample_size = 1
    project.target_count = 100
    monkeypatch.setattr(generate, 'filter_druglike', lambda rows: [
        dict(row, druglike_selected=False, druglike_reject_reason='qed_rank') for row in rows])
    monkeypatch.setattr(combichem, '_next_attachment_selection', lambda *a: pytest.fail('changed supplied core'))
    with pytest.raises(RuntimeError, match='combinations exhausted'):
        generate._run_stage(project, 1)
    assert database.combichem_empty_batches(project.db_path, 1) == 10
    assert database.iteration_scaffold(project.db_path, 1)['source'] == 'user_elite_mcs'


def test_attachment_switch_requires_ten_batches_and_archives_partial_round(adaptive_project):
    args = adaptive_project
    original = database.iteration_scaffold(args.db_path, 2)
    for i in range(9):
        database.insert_generated_candidates(args.db_path, 2, 100, [], combichem_progress=(i, i + 1, 0))
    before = database.combichem_plan(args.db_path, 2)
    with pytest.raises(RuntimeError, match='10 consecutive'):
        combichem.refill_plan(args, 2, args.output / 'iter2/combichem', before, change_attachments=True)
    assert database.combichem_plan(args.db_path, 2) == before
    database.insert_generated_candidates(args.db_path, 2, 100, [], combichem_progress=(9, 10, 0))
    before = database.combichem_plan(args.db_path, 2)
    switched = combichem.refill_plan(args, 2, args.output / 'iter2/combichem', before, change_attachments=True)
    assert switched['round'] == switched['attachment_revision'] == 1
    assert switched['next_index'] == 0
    assert switched['attachment_selection']['replacement_cuts'] != original['replacement_cuts']
    assert database.combichem_empty_batches(args.db_path, 2, 1) == 0
    assert database.iteration_scaffold(args.db_path, 2) == original
    with database.connect(args.db_path) as conn:
        archived, cursor = conn.execute('SELECT plan,next_index FROM iteration_combichem_history WHERE iteration=2 AND round=0').fetchone()
    assert dict(json.loads(archived), next_index=int(cursor)) == before
    assert int(cursor) == 10 < int(before['total_combinations'])
    with pytest.raises(RuntimeError, match='batch round'):
        database.insert_generated_candidates(args.db_path, 2, 100, [], combichem_progress=(10, 11, 0))
    args.sidechain_library.unlink()
    assert combichem.prepare_plan(args, 2, args.output / 'iter2/combichem') == switched


def test_concurrent_attachment_switches_install_only_one_round(adaptive_project):
    args = adaptive_project
    for i in range(10):
        database.insert_generated_candidates(args.db_path, 2, 100, [], combichem_progress=(i, i + 1, 0))
    previous = database.combichem_plan(args.db_path, 2)
    def switch(_):
        return combichem.refill_plan(args, 2, args.output / 'iter2/combichem', previous, change_attachments=True)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(switch, range(2)))
    assert results[0] == results[1] and results[0]['round'] == 1
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT count(*) FROM iteration_combichem_history WHERE iteration=2').fetchone()[0] == 1


def test_generation_switches_after_tenth_empty_batch_and_restores_compound_origin(adaptive_project, monkeypatch):
    args = adaptive_project
    with database.connect(args.db_path) as conn:
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,gbsa_score) VALUES ('preserved','CCO',2,'gbsa_done',-40)")
    original_filter = generate.filter_druglike
    calls = []
    def reject(rows):
        calls.append(len(rows))
        return [dict(row, druglike_selected=False, druglike_reject_reason='qed_rank') for row in rows]
    monkeypatch.setattr(generate, 'filter_druglike', reject)
    original_refill = combichem.refill_plan
    def interrupt_after_switch(*a, **kw):
        result = original_refill(*a, **kw)
        assert kw.get('change_attachments') and result['attachment_revision'] == 1
        raise RuntimeError('attachment switch committed')
    monkeypatch.setattr(combichem, 'refill_plan', interrupt_after_switch)
    with pytest.raises(RuntimeError, match='attachment switch committed'):
        generate._run_combichem(args, 2, args.output / 'iter2/combichem')
    assert len(calls) == 10
    active = database.combichem_plan(args.db_path, 2)
    args.target_count = 2
    args.sidechain_library.unlink()
    monkeypatch.setattr(generate, 'filter_druglike', original_filter)
    monkeypatch.setattr(combichem, 'refill_plan', original_refill)
    generate._run_combichem(args, 2, args.output / 'iter2/combichem')
    with database.connect(args.db_path) as conn:
        assert conn.execute("SELECT state,gbsa_score FROM compound WHERE name='preserved'").fetchone() == ('gbsa_done', -40)
        cid, name, smiles = conn.execute("SELECT id,name,smiles FROM compound WHERE iteration=2 AND name<>'preserved'").fetchone()
    assert database.combichem_compound_scaffold(args.db_path, cid) == active['attachment_selection']
    # The origin survives archiving its producing round and is used for N+1.
    current = database.combichem_plan(args.db_path, 2)
    database.insert_generated_candidates(args.db_path, 2, 100, [],
        combichem_progress=(current['next_index'], int(current['total_combinations']), current['round']))
    current = database.combichem_plan(args.db_path, 2)
    database.advance_combichem_plan(args.db_path, 2, current, current, 100)
    assert database.combichem_compound_scaffold(args.db_path, cid) == active['attachment_selection']
    helper = combichem.prepare_single_elite_scaffold
    seen = []
    def capture(elite, previous, *a, **kw):
        seen.append(previous)
        return helper(elite, previous, *a, **kw)
    monkeypatch.setattr(combichem, 'prepare_single_elite_scaffold', capture)
    prepare_iteration_scaffold(args, 3, [(cid, name, smiles, 2, -10, -40)], create=True)
    assert seen == [active['attachment_selection']]


def test_untried_attachment_combinations_are_not_repeated_and_exhaustion_stops(adaptive_project):
    args = adaptive_project
    snapshot = database.iteration_scaffold(args.db_path, 2)
    tried = {tuple(cut['bond_index'] for cut in snapshot['replacement_cuts'])}
    total = math.prod(len(group) for group in snapshot['replacement_candidates'])
    for _ in range(total - 1):
        current = database.combichem_plan(args.db_path, 2)
        for i in range(10):
            database.insert_generated_candidates(args.db_path, 2, 100, [],
                combichem_progress=(i, i + 1, current.get('round', 0)))
        current = database.combichem_plan(args.db_path, 2)
        new = combichem.refill_plan(args, 2, args.output / 'iter2/combichem', current, change_attachments=True)
        signature = tuple(cut['bond_index'] for cut in new['attachment_selection']['replacement_cuts'])
        assert signature not in tried
        tried.add(signature)
    assert len(tried) == total
    with pytest.raises(RuntimeError, match='All single-elite attachment combinations'):
        combichem._next_attachment_selection(args, 2)


def test_combichem_generation_uses_existing_filters_target_and_scaffold(project, monkeypatch):
    monkeypatch.setattr(combichem.secrets, 'randbits', lambda _: 123)
    generate._run_stage(project, 1)
    snapshot = database.iteration_scaffold(project.db_path, 1)
    plan = database.combichem_plan(project.db_path, 1)
    assert snapshot['generation_method'] == 'combichem'
    assert plan['total_combinations'] == '16'
    assert 0 < plan['next_index'] < int(plan['total_combinations'])
    assert math.gcd(int(plan['stride']), 16) == 1
    with database.connect(project.db_path) as conn:
        rows = conn.execute('SELECT smiles,state,rdk_qed,rdk_ro5_violations FROM compound WHERE iteration=1').fetchall()
        assert len(rows) == len({row[0] for row in rows}) == 4
        assert sum(row[1] == 'sa_rejected' for row in rows) == 1
        assert all(row[2] is not None and row[3] <= 1 and '*' not in row[0] for row in rows)
    assert list((project.output / 'iter1/combichem').glob('assembly-*.tsv'))
    assert list((project.output / 'iter1/combichem').glob('druglike_selection-*.csv'))
    generate._run_stage(project, 1)
    assert database.combichem_plan(project.db_path, 1) == plan
    assert database.iteration_scaffold(project.db_path, 1) == snapshot


def test_saved_pools_and_parameters_survive_library_and_config_changes(project, monkeypatch):
    prepare_iteration_scaffold(project, 1, create=True)
    directory = project.output / 'iter1/combichem'
    monkeypatch.setattr(combichem.secrets, 'randbits', lambda _: 123)
    first = combichem.prepare_plan(project, 1, directory)
    assert int(first['seed']) == 123
    original = [combichem.combination_at(first, i) for i in range(16)]
    project.sidechain_library.unlink()
    project.sample_size = 500
    monkeypatch.setattr(combichem, 'load_library', lambda *a: pytest.fail('reread saved library'))
    restored = combichem.prepare_plan(project, 1, directory)
    assert restored == first
    assert restored['batch_size'] == 4
    assert [combichem.combination_at(restored, i) for i in range(16)] == original


def test_combichem_exhaustion_does_not_complete_underfilled_iteration(project):
    project.target_count = 100
    for _ in range(2):
        with pytest.raises(RuntimeError, match='combinations exhausted'):
            generate._run_stage(project, 1)
        plan = database.combichem_plan(project.db_path, 1)
        assert plan['next_index'] == int(plan['total_combinations'])
        assert database.stage_completion_reason(project.db_path, 1, 'generate', 100, 2) is not None
        assert database.iteration_compound_count(project.db_path, 1) < 100
    directory = project.output / 'iter1/combichem'
    attempts = [row for audit in directory.glob('assembly-*.tsv')
                for row in csv.DictReader(audit.open(), delimiter='\t')]
    assert len(attempts) == len({row['combination'] for row in attempts}) == 16
    assert len({row['sidechain_lines'] for row in attempts}) == 16


def test_insertion_and_combichem_cursor_commit_together_and_replay_is_idempotent(project):
    database.combichem_plan(project.db_path, 1, dict(total_combinations='4'))
    candidate = dict(calculate_rdkit_descriptors('CCO'), smiles='CCO', druglike_selected=True)
    assert database.insert_generated_candidates(project.db_path, 1, 4, [candidate],
                                                combichem_progress=(0, 1)) == 1
    another = dict(calculate_rdkit_descriptors('CCN'), smiles='CCN', druglike_selected=True)
    assert database.insert_generated_candidates(project.db_path, 1, 4, [another],
                                                combichem_progress=(0, 1)) == 0
    assert database.iteration_compound_count(project.db_path, 1) == 1
    bad = dict(calculate_rdkit_descriptors('CCC'), smiles='CCC',
               druglike_selected=True, rdk_qed=float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        database.insert_generated_candidates(project.db_path, 1, 4, [another, bad],
                                              combichem_progress=(1, 2))
    assert database.combichem_plan(project.db_path, 1)['next_index'] == 1
    assert database.iteration_compound_count(project.db_path, 1) == 1


def test_restart_after_result_commit_uses_next_batch(project, monkeypatch):
    original = generate.insert_druglike_batch
    calls = []
    def crash_after_commit(*args, **kwargs):
        inserted = original(*args, **kwargs)
        calls.append(kwargs['combichem_progress'])
        raise RuntimeError('simulated crash after commit')
    monkeypatch.setattr(generate, 'insert_druglike_batch', crash_after_commit)
    with pytest.raises(RuntimeError, match='simulated crash'):
        generate._run_stage(project, 1)
    assert calls == [(0, 4, 0)]
    before = database.combichem_plan(project.db_path, 1)
    assert before['next_index'] == 4
    monkeypatch.setattr(generate, 'insert_druglike_batch', original)
    generate._run_stage(project, 1)
    after = database.combichem_plan(project.db_path, 1)
    assert after['seed'] == before['seed']
    assert after['pools'] == before['pools']
    assert after['next_index'] > before['next_index']
    assert database.iteration_compound_count(project.db_path, 1) == 4


def test_preceding_iteration_gate_still_blocks_combichem(project):
    with database.connect(project.db_path) as conn:
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('old','CCO',1,'generated')")
    generate._run_stage(project, 2)
    assert database.iteration_scaffold(project.db_path, 2) is None
    assert database.combichem_plan(project.db_path, 2) is None


def test_generator_switch_cannot_replace_a_saved_iteration(project):
    prepare_iteration_scaffold(project, 1, create=True)
    saved = database.iteration_scaffold(project.db_path, 1)
    project.sidechain_library = None
    with pytest.raises(RuntimeError, match='cannot switch generator'):
        prepare_iteration_scaffold(project, 1, create=True)
    assert database.iteration_scaffold(project.db_path, 1) == saved
    # Downstream stages restore the recorded core regardless of generator config.
    prepare_iteration_scaffold(project, 1, include_generation=False)


def test_descriptor_failure_retains_cursor_pools_and_scientific_rows(project, monkeypatch):
    original = generate.calculate_rdkit_descriptors_batch
    def broken(*a, **k):
        raise RuntimeError('descriptor failure')
    monkeypatch.setattr(generate, 'calculate_rdkit_descriptors_batch', broken)
    with pytest.raises(RuntimeError, match='descriptor failure'):
        generate._run_stage(project, 1)
    before = database.combichem_plan(project.db_path, 1)
    assert before['next_index'] == 0
    assert database.iteration_compound_count(project.db_path, 1) == 0
    monkeypatch.setattr(generate, 'calculate_rdkit_descriptors_batch', original)
    generate._run_stage(project, 1)
    after = database.combichem_plan(project.db_path, 1)
    assert after['seed'] == before['seed'] and after['pools'] == before['pools']


def test_zero_accepted_batch_advances_and_continues(project, monkeypatch):
    original = generate.filter_druglike
    calls = []
    def first_empty(rows):
        selected = original(rows)
        calls.append(len(rows))
        if len(calls) == 1:
            for row in selected:
                row.update(druglike_selected=False, druglike_reject_reason='qed_rank')
        return selected
    monkeypatch.setattr(generate, 'filter_druglike', first_empty)
    generate._run_stage(project, 1)
    assert len(calls) > 1
    assert database.iteration_compound_count(project.db_path, 1) == 4
    assert database.combichem_plan(project.db_path, 1)['next_index'] >= 8


def test_concurrent_batch_commit_does_not_duplicate_products_or_advance_twice(project):
    database.combichem_plan(project.db_path, 1, dict(total_combinations='1000000000000000000000000'))
    row = dict(calculate_rdkit_descriptors('CCO'), smiles='CCO', druglike_selected=True)
    def persist(_):
        return database.insert_generated_candidates(project.db_path, 1, 4, [row],
                                                    combichem_progress=(0, 4))
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(persist, range(2))) == [0, 1]
    assert database.combichem_plan(project.db_path, 1)['next_index'] == 4
    assert database.iteration_compound_count(project.db_path, 1) == 1
    with pytest.raises(RuntimeError, match='cursor'):
        database.insert_generated_candidates(project.db_path, 1, 4, [], combichem_progress=(0, 8))


def test_refill_resumes_after_install_crash_and_preserves_existing_compounds(project, monkeypatch):
    monkeypatch.setattr(combichem, 'select_sidechains', lambda library, rng: [dict(library[0])])
    original_insert = generate.insert_druglike_batch
    def stop_after_first_commit(*args, **kwargs):
        original_insert(*args, **kwargs)
        raise RuntimeError('first round committed')
    monkeypatch.setattr(generate, 'insert_druglike_batch', stop_after_first_commit)
    with pytest.raises(RuntimeError, match='first round committed'):
        generate._run_stage(project, 1)
    previous = database.combichem_plan(project.db_path, 1)
    assert previous['next_index'] == int(previous['total_combinations']) == 1
    assert database.iteration_compound_count(project.db_path, 1) == 1
    snapshot = database.iteration_scaffold(project.db_path, 1)
    with database.connect(project.db_path) as conn:
        original_compound = conn.execute('SELECT id,name,smiles,rdk_sa_score FROM compound WHERE iteration=1').fetchone()
    original_refill = combichem.refill_plan
    def stop_after_refill(*args, **kwargs):
        original_refill(*args, **kwargs)
        raise RuntimeError('new pool committed')
    monkeypatch.setattr(generate, 'insert_druglike_batch', original_insert)
    monkeypatch.setattr(combichem, 'select_sidechains', lambda library, rng: [dict(row) for row in library])
    monkeypatch.setattr(combichem, 'refill_plan', stop_after_refill)
    with pytest.raises(RuntimeError, match='new pool committed'):
        generate._run_stage(project, 1)
    active = database.combichem_plan(project.db_path, 1)
    assert active['round'] == 1 and active['next_index'] == 0
    assert database.iteration_compound_count(project.db_path, 1) == 1
    with database.connect(project.db_path) as conn:
        archived, cursor = conn.execute('SELECT plan,next_index FROM iteration_combichem_history '
                                        'WHERE iteration=1 AND round=0').fetchone()
    assert dict(json.loads(archived), next_index=int(cursor)) == previous
    # A restart after pool commit needs neither the library nor a new seed.
    project.sidechain_library.unlink()
    monkeypatch.setattr(combichem, 'refill_plan', original_refill)
    monkeypatch.setattr(combichem, 'load_library', lambda *a: pytest.fail('reread committed refill library'))
    monkeypatch.setattr(combichem.secrets, 'randbits', lambda *a: pytest.fail('reseeded committed refill'))
    generate._run_stage(project, 1)
    final = database.combichem_plan(project.db_path, 1)
    assert final['round'] == 1 and final['seed'] == active['seed'] and final['pools'] == active['pools']
    assert database.iteration_compound_count(project.db_path, 1) == 4
    assert database.iteration_scaffold(project.db_path, 1) == snapshot
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT id,name,smiles,rdk_sa_score FROM compound WHERE id=?',
                            (original_compound[0],)).fetchone() == original_compound
        assert conn.execute('SELECT MAX(rdk_ro5_violations) FROM compound WHERE iteration=1').fetchone()[0] <= 1
    root = project.output / 'iter1/combichem'
    assert (root / 'generated-0-1.csv').exists()
    assert list((root / 'round-000001').glob('generated-*.csv'))


def test_legacy_refill_is_atomic_and_old_round_batches_cannot_replay(project, monkeypatch):
    prepare_iteration_scaffold(project, 1, create=True)
    directory = project.output / 'iter1/combichem'
    monkeypatch.setattr(combichem, 'select_sidechains', lambda library, rng: [dict(library[0])])
    saved = combichem.prepare_plan(project, 1, directory)
    # Represent an installed legacy plan with no round metadata.
    legacy = {key: value for key, value in saved.items() if key not in ('round', 'next_index')}
    with database.connect(project.db_path) as conn:
        conn.execute('UPDATE iteration_combichem SET plan=? WHERE iteration=1', (json.dumps(legacy),))
    database.insert_generated_candidates(project.db_path, 1, 4, [], combichem_progress=(0, 1))
    previous = database.combichem_plan(project.db_path, 1)
    following = dict(legacy, seed='different')
    def refill(_):
        return database.advance_combichem_plan(project.db_path, 1, previous, following, 4)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(refill, range(2)))
    assert results[0] == results[1]
    assert results[0]['round'] == 1 and results[0]['next_index'] == 0
    with database.connect(project.db_path) as conn:
        rows = conn.execute('SELECT round,plan,next_index FROM iteration_combichem_history WHERE iteration=1').fetchall()
    assert len(rows) == 1 and rows[0][0] == 0
    assert json.loads(rows[0][1]) == legacy and rows[0][2] == '1'
    for progress in ((0, 1), (0, 1, 0)):
        with pytest.raises(RuntimeError, match='batch round'):
            database.insert_generated_candidates(project.db_path, 1, 4, [], combichem_progress=progress)
    database.insert_generated_candidates(project.db_path, 1, 4, [], combichem_progress=(0, 1, 1))
    assert database.combichem_plan(project.db_path, 1)['next_index'] == 1
    assert database.insert_generated_candidates(project.db_path, 1, 4, [], combichem_progress=(0, 1, 1)) == 0


@pytest.mark.parametrize('change', ['modified', 'missing'])
def test_refill_library_failure_preserves_exhausted_plan_and_results(project, monkeypatch, change):
    prepare_iteration_scaffold(project, 1, create=True)
    directory = project.output / 'iter1/combichem'
    monkeypatch.setattr(combichem, 'select_sidechains', lambda library, rng: [dict(library[0])])
    combichem.prepare_plan(project, 1, directory)
    candidate = dict(calculate_rdkit_descriptors('CCO'), smiles='CCO', druglike_selected=True)
    database.insert_generated_candidates(project.db_path, 1, 4, [candidate], combichem_progress=(0, 1, 0))
    previous = database.combichem_plan(project.db_path, 1)
    if change == 'modified':
        project.sidechain_library.write_text('*CCC\n')
    else:
        project.sidechain_library.unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        combichem.refill_plan(project, 1, directory, previous)
    assert database.combichem_plan(project.db_path, 1) == previous
    assert database.iteration_compound_count(project.db_path, 1) == 1


def test_refill_requires_exhaustion_and_keeps_saved_parameters(project, monkeypatch):
    prepare_iteration_scaffold(project, 1, create=True)
    directory = project.output / 'iter1/combichem'
    monkeypatch.setattr(combichem, 'select_sidechains', lambda library, rng: [dict(library[0])])
    previous = combichem.prepare_plan(project, 1, directory)
    following = {key: value for key, value in previous.items() if key != 'next_index'}
    with pytest.raises(RuntimeError, match='unfinished'):
        database.advance_combichem_plan(project.db_path, 1, previous, following, 4)
    database.insert_generated_candidates(project.db_path, 1, 4, [], combichem_progress=(0, 1, 0))
    previous = database.combichem_plan(project.db_path, 1)
    with pytest.raises(RuntimeError, match='differs'):
        database.advance_combichem_plan(project.db_path, 1, previous, dict(following, batch_size=999), 4)
    project.sample_size = 999
    project.sidechain_library = project.output / 'different-library.smi'
    following = combichem.refill_plan(project, 1, directory, previous)
    assert following['round'] == 1
    assert following['batch_size'] == previous['batch_size']
    assert following['library_path'] == previous['library_path']


def test_refill_rolls_back_archive_if_new_plan_cannot_commit(project, monkeypatch):
    prepare_iteration_scaffold(project, 1, create=True)
    directory = project.output / 'iter1/combichem'
    monkeypatch.setattr(combichem, 'select_sidechains', lambda library, rng: [dict(library[0])])
    combichem.prepare_plan(project, 1, directory)
    database.insert_generated_candidates(project.db_path, 1, 4, [], combichem_progress=(0, 1, 0))
    previous = database.combichem_plan(project.db_path, 1)
    with database.connect(project.db_path) as conn:
        conn.execute("CREATE TRIGGER fail_refill BEFORE UPDATE OF plan ON iteration_combichem "
                     "BEGIN SELECT RAISE(ABORT, 'simulated refill failure'); END")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError, match='simulated refill failure'):
        combichem.refill_plan(project, 1, directory, previous)
    assert database.combichem_plan(project.db_path, 1) == previous
    with database.connect(project.db_path) as conn:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_combichem_history'").fetchone()
        if exists:
            assert conn.execute('SELECT count(*) FROM iteration_combichem_history').fetchone()[0] == 0
        conn.execute('DROP TRIGGER fail_refill')
    assert combichem.refill_plan(project, 1, directory, previous)['round'] == 1


def test_refill_cannot_install_new_round_after_target_is_met(project, monkeypatch):
    prepare_iteration_scaffold(project, 1, create=True)
    directory = project.output / 'iter1/combichem'
    monkeypatch.setattr(combichem, 'select_sidechains', lambda library, rng: [dict(library[0])])
    combichem.prepare_plan(project, 1, directory)
    candidate = dict(calculate_rdkit_descriptors('CCO'), smiles='CCO', druglike_selected=True)
    database.insert_generated_candidates(project.db_path, 1, 1, [candidate], combichem_progress=(0, 1, 0))
    previous = database.combichem_plan(project.db_path, 1)
    following = {key: value for key, value in previous.items() if key != 'next_index'}
    assert database.advance_combichem_plan(project.db_path, 1, previous, following, 1) == previous
    assert database.combichem_plan(project.db_path, 1) == previous


def test_new_iteration_uses_only_immediately_preceding_elites_without_tl(project, monkeypatch):
    generate._run_stage(project, 1)
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state='gbsa_done',docking_score=-10,gbsa_score=-id WHERE iteration=1")
    selected = database.iteration_gbsa_elites(project.db_path, 1, 2)
    seen = []
    monkeypatch.setattr(generate, '_run_combichem', lambda args, iteration, directory: seen.append(iteration))
    generate._run_stage(project, 2)
    snapshot = database.iteration_scaffold(project.db_path, 2)
    assert seen == [2]
    assert snapshot['source_ids'] == [row[0] for row in selected]
    assert snapshot['generation_method'] == 'combichem'
    assert snapshot['docking_mcs_smarts'] == 'c1ccccc1'


def test_zero_attachment_combichem_core_reuses_previous_scaffold(project):
    prepare_iteration_scaffold(project, 1, create=True)
    elites = [(100, 'A', 'CCc1ccc(CC)cc1', 1, -10, -40),
              (101, 'B', 'CCc1ccc(CC)cc1', 1, -9, -30)]
    prepare_iteration_scaffold(project, 2, elites, create=True)
    snapshot = database.iteration_scaffold(project.db_path, 2)
    assert snapshot['source'] == 'previous_iteration_scaffold_fallback'
    assert snapshot['generation_method'] == 'combichem'
    assert snapshot['fallback_from_iteration'] == 1


@pytest.mark.parametrize('smiles', ['CCc1ccc(CC)cc1', 'C[C@H](F)c1ccc(CC)cc1',
                                    'c1ccc(-c2ccc(CC)cc2)cc1'])
def test_single_elite_replaces_distal_sidechain_parts_and_preserves_core(smiles, monkeypatch):
    previous = dict(mcs_smarts='c1ccccc1', scaffold_smiles='[*:0]c1ccc([*:1])cc1')
    monkeypatch.setattr(combichem.secrets, 'choice', lambda candidates: candidates[0])
    info = combichem.prepare_single_elite_scaffold(dict(id=100, mol=Chem.MolFromSmiles(smiles)),
                                               previous, 'c1ccccc1')
    original = Chem.MolFromSmiles(info['replacement_elite_smiles'])
    query = Chem.MolFromSmarts(previous['scaffold_smiles'])
    match = original.GetSubstructMatch(query)
    old_core = {index for atom, index in zip(query.GetAtoms(), match) if atom.GetAtomicNum() != 0}
    groups = info['replacement_candidates']
    removed = set()
    for label, cut in enumerate(info['replacement_cuts']):
        assert cut in groups[label]
        bond = original.GetBondWithIdx(cut['bond_index'])
        assert bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing()
        assert {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} == {cut['retained_atom_index'], cut['distal_atom_index']}
        deleted = set(cut['removed_atom_indices'])
        assert deleted and not (deleted & old_core) and not (deleted & removed)
        assert cut['distal_atom_index'] in deleted and cut['retained_atom_index'] not in deleted
        removed.update(deleted)
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
    assert all(atom.GetAtomicNum() != 0 for atom in Chem.MolFromSmarts(info['mcs_smarts']).GetAtoms())
    assert len(combichem.attachment_atoms(scaffold)) == 2
    assert '[*:0]' in info['scaffold_smiles'] and '[*:1]' in info['scaffold_smiles']
    assert scaffold.GetNumHeavyAtoms() == original.GetNumHeavyAtoms() - len(removed)
    assert scaffold.HasSubstructMatch(Chem.MolFromSmarts(previous['mcs_smarts']))
    product = combichem.assemble(scaffold, [Chem.MolFromSmiles('*C')] * 2)
    assert product is not None
    assert Chem.MolFromSmiles(product).GetNumHeavyAtoms() == scaffold.GetNumHeavyAtoms() + 2


def test_single_elite_replacement_can_replace_whole_sidechains(monkeypatch):
    original = Chem.MolFromSmiles('CCCCc1ccc(CCCCC)cc1')
    def choose(candidates):
        return max(candidates, key=lambda cut: len(cut['removed_atom_indices']))
    monkeypatch.setattr(combichem.secrets, 'choice', choose)
    info = combichem.prepare_single_elite_scaffold(dict(id=100, mol=original),
        dict(mcs_smarts='c1ccccc1', scaffold_smiles='[*:0]c1ccc([*:1])cc1'), 'c1ccccc1')
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
    assert scaffold.GetNumHeavyAtoms() == 6
    assert combichem.assemble(scaffold, [Chem.MolFromSmiles('*C')] * 2) == 'Cc1ccc(C)cc1'


@pytest.mark.parametrize('removed_count, replacement', [(1, '*C'), (2, '*CC')])
def test_single_elite_replacement_preserves_retained_stereocenter_and_charge(monkeypatch, removed_count, replacement):
    original = Chem.MolFromSmiles('CC[C@H]([NH3+])c1ccc([13CH3])cc1')
    def choose(candidates):
        return next(cut for cut in candidates
                    if len(cut['removed_atom_indices']) == removed_count
                    and original.GetAtomWithIdx(cut['distal_atom_index']).GetAtomicNum() == 6)
    # Helper indexes are based on canonical SMILES.
    original = Chem.MolFromSmiles(Chem.MolToSmiles(original))
    monkeypatch.setattr(combichem.secrets, 'choice', choose)
    info = combichem.prepare_single_elite_scaffold(dict(id=100, mol=original),
        dict(mcs_smarts='c1ccc([13CH3])cc1', scaffold_smiles='[*:0]c1ccc([13CH3])cc1'), 'c1ccccc1')
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
    assert Chem.GetFormalCharge(scaffold) == 1
    assert any(atom.GetIsotope() == 13 for atom in scaffold.GetAtoms())
    assert combichem.assemble(scaffold, [Chem.MolFromSmiles(replacement)]) == info['replacement_elite_smiles']


def test_single_elite_replacement_preserves_retained_double_bond_stereo(monkeypatch):
    original = Chem.MolFromSmiles('F/C=C/CCc1ccccc1')
    original = Chem.MolFromSmiles(Chem.MolToSmiles(original))
    monkeypatch.setattr(combichem.secrets, 'choice', lambda candidates: next(
        cut for cut in candidates if original.GetAtomWithIdx(cut['distal_atom_index']).GetAtomicNum() == 9))
    info = combichem.prepare_single_elite_scaffold(dict(id=100, mol=original),
        dict(mcs_smarts='c1ccccc1', scaffold_smiles='[*:0]c1ccccc1'), 'c1ccccc1')
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
    assert combichem.assemble(scaffold, [Chem.MolFromSmiles('*F')]) == info['replacement_elite_smiles']


def test_single_elite_replacement_preserves_stereo_with_two_cuts_at_same_atom(monkeypatch):
    monkeypatch.setattr(combichem.secrets, 'choice',
                        lambda candidates: max(candidates, key=lambda cut: len(cut['removed_atom_indices'])))
    info = combichem.prepare_single_elite_scaffold(
        dict(id=100, mol=Chem.MolFromSmiles('N[C@H](CC)CCC')),
        dict(mcs_smarts='NC', scaffold_smiles='[*:0][C@H](N)[*:1]'), 'NC',
    )
    assert len({cut['retained_atom_index'] for cut in info['replacement_cuts']}) == 1
    originals = [Chem.MolFromSmiles('*' + 'C' * len(cut['removed_atom_indices']))
                 for cut in info['replacement_cuts']]
    assert combichem.assemble(Chem.MolFromSmiles(info['scaffold_smiles']), originals) == info['replacement_elite_smiles']


def test_single_elite_replacement_removes_whole_rings_without_splitting_them(monkeypatch):
    original = Chem.MolFromSmiles('c1ccc(-c2ccccc2)cc1')
    monkeypatch.setattr(combichem.secrets, 'choice', lambda candidates: candidates[0])
    info = combichem.prepare_single_elite_scaffold(dict(id=100, mol=original),
        dict(mcs_smarts='c1ccccc1', scaffold_smiles='[*:0]c1ccccc1'), 'c1ccccc1')
    assert len(info['replacement_candidates'][0]) == 1
    assert len(info['replacement_cuts'][0]['removed_atom_indices']) == 6
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
    assert scaffold.GetNumHeavyAtoms() == 6
    assert combichem.assemble(scaffold, [Chem.MolFromSmiles('*Cl')]) == 'Clc1ccccc1'


@pytest.mark.parametrize('smiles', ['Clc1ccc(F)cc1', 'Oc1ccc(N)cc1', 'FC(F)(F)c1ccc(Cl)cc1'])
def test_single_elite_replacement_needs_no_sidechain_hydrogen(smiles):
    previous = dict(mcs_smarts='c1ccccc1', scaffold_smiles='[*:0]c1ccc([*:1])cc1')
    info = combichem.prepare_single_elite_scaffold(dict(id=100, mol=Chem.MolFromSmiles(smiles)),
                                               previous, 'c1ccccc1')
    assert len(combichem.attachment_atoms(Chem.MolFromSmiles(info['scaffold_smiles']))) == 2


def test_legacy_literal_dummy_query_uses_saved_retained_core_without_rewriting_it():
    smiles = '[*:0]CCc1ccc(C[*:1])cc1'
    previous = dict(source='single_elite_sidechain_replacement', scaffold_smiles=smiles,
                    mcs_smarts=Chem.MolToSmarts(Chem.MolFromSmiles(smiles)))
    before = dict(previous)
    info = combichem.prepare_single_elite_scaffold(
        dict(id=100, mol=Chem.MolFromSmiles('CCCCc1ccc(CCC)cc1')), previous, 'c1ccccc1')
    query = Chem.MolFromSmarts(info['mcs_smarts'])
    scaffold = Chem.MolFromSmiles(info['scaffold_smiles'])
    assert query.GetNumAtoms() == scaffold.GetNumHeavyAtoms()
    assert all(atom.GetAtomicNum() != 0 for atom in query.GetAtoms())
    assert previous == before


@pytest.mark.parametrize('scaffold, smiles, count', [
    ('[*:0]c1ccccc1', 'CCc1ccccc1', 1),
    ('[*:0]c1ccc([*:1])cc1', 'CCc1ccc(CC)cc1', 2),
    ('[*:0]c1cc([*:1])cc([*:2])c1', 'CCc1cc(CC)cc(CC)c1', 3),
])
def test_single_elite_keeps_preceding_attachment_count(scaffold, smiles, count):
    info = combichem.prepare_single_elite_scaffold(
        dict(id=100, mol=Chem.MolFromSmiles(smiles)),
        dict(mcs_smarts='c1ccccc1', scaffold_smiles=scaffold), 'c1ccccc1',
    )
    mol = Chem.MolFromSmiles(info['scaffold_smiles'])
    labels = [mol.GetAtomWithIdx(index).GetAtomMapNum() for index in combichem.attachment_atoms(mol)]
    assert sorted(labels) == list(range(count))
    indices = [cut['bond_index'] for cut in info['replacement_cuts']]
    assert len(indices) == len(set(indices)) == count
    assert combichem.assemble(mol, [Chem.MolFromSmiles('*C')] * count) is not None


def test_single_elite_rejects_sidechain_with_only_multiple_bond_cuts():
    mol = Chem.MolFromSmiles('O=C1CCCCC1')
    assert combichem._terminal_cuts(mol, {0}, set(range(1, mol.GetNumAtoms()))) == []
    with pytest.raises(RuntimeError, match='incompatible'):
        combichem.prepare_single_elite_scaffold(
            dict(id=100, mol=Chem.MolFromSmiles('O=C1CCCCC1')),
            dict(mcs_smarts='C1CCCCC1', scaffold_smiles='[*:0]C1CCCCC1'), 'C1CCCCC1',
        )


def test_terminal_cut_candidates_do_not_treat_isotopic_hydrogen_as_a_sidechain():
    mol = Chem.MolFromSmiles('[2H]Cc1ccccc1')
    assert combichem._terminal_cuts(mol, {0}, set(range(1, mol.GetNumAtoms()))) == []


def test_saved_single_attachment_snapshot_is_not_rewritten(project, monkeypatch):
    prepare_iteration_scaffold(project, 1, create=True)
    legacy = dict(mcs_smarts='c1ccccc1', scaffold_smiles='[*:0]CCc1ccc(CC)cc1',
                  source='single_elite_sidechain_expansion', expansion_from_iteration=1,
                  expansion_atom_index=0, expansion_candidate_indices=[0], source_ids=[100],
                  docking_mcs_smarts='c1ccccc1', generation_method='combichem')
    database.iteration_scaffold(project.db_path, 2, legacy)
    monkeypatch.setattr(combichem.secrets, 'choice', lambda _: pytest.fail('changed saved expansion'))
    prepare_iteration_scaffold(project, 2, create=True)
    assert database.iteration_scaffold(project.db_path, 2) == legacy


def test_single_elite_snapshot_survives_restart_without_reselecting(project, monkeypatch, capsys):
    prepare_iteration_scaffold(project, 1, create=True)
    original = database.iteration_scaffold(project.db_path, 1)
    project.gbsa_elite_count = 1
    project.elite_mcs_smarts = None
    elite = [(100, 'A', 'CCc1ccc(CC)cc1', 1, -10, -40)]
    monkeypatch.setattr(combichem.secrets, 'choice', lambda candidates: candidates[-1])
    prepare_iteration_scaffold(project, 2, elite, create=True)
    saved = database.iteration_scaffold(project.db_path, 2)
    assert saved['source'] == 'single_elite_sidechain_replacement'
    assert saved['source_ids'] == [100]
    assert saved['replacement_from_iteration'] == 1
    assert saved['replacement_cuts'] == [group[-1] for group in saved['replacement_candidates']]
    assert len(saved['replacement_cuts']) == 2
    assert saved['docking_mcs_smarts'] == original['docking_mcs_smarts']
    monkeypatch.setattr(combichem.secrets, 'choice', lambda _: pytest.fail('reselected expansion site'))
    prepare_iteration_scaffold(project, 2, create=True)
    assert database.iteration_scaffold(project.db_path, 2) == saved
    assert database.iteration_scaffold(project.db_path, 1) == original
    assert 'single elite terminal sidechain replacement from iteration 1' in capsys.readouterr().out


@pytest.mark.parametrize('previous', [None, dict(mcs_smarts='c1ccccc1',
                                               scaffold_smiles='[*:0]C1CCCCC1')])
def test_single_elite_rejects_missing_or_incompatible_previous_scaffold(previous):
    with pytest.raises(RuntimeError):
        combichem.prepare_single_elite_scaffold(dict(id=100, mol=Chem.MolFromSmiles('CCc1ccc(CC)cc1')),
                                               previous, 'c1ccccc1')


def test_single_elite_generation_preserves_completion_gate_and_skips_tl(project, monkeypatch):
    generate._run_stage(project, 1)
    project.gbsa_elite_count = 1
    project.elite_mcs_smarts = None
    started = []
    monkeypatch.setattr(generate, '_run_combichem', lambda *args: started.append(args[1]))
    generate._run_stage(project, 2)
    assert not started
    assert database.iteration_scaffold(project.db_path, 2) is None
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state='gbsa_done',docking_score=-10,gbsa_score=-id WHERE iteration=1")
    elite = database.iteration_gbsa_elites(project.db_path, 1, 1)[0]
    generate._run_stage(project, 2)
    assert started == [2]
    saved = database.iteration_scaffold(project.db_path, 2)
    assert saved['source'] == 'single_elite_sidechain_replacement'
    assert saved['source_ids'] == [elite[0]]


def test_single_elite_does_not_bypass_configured_elite_count(project):
    prepare_iteration_scaffold(project, 1, create=True)
    project.elite_mcs_smarts = None
    with pytest.raises(RuntimeError, match='selected preceding-iteration elites'):
        prepare_iteration_scaffold(project, 2, [(100, 'A', 'CCc1ccc(CC)cc1', 1, -10, -40)], create=True)
    assert database.iteration_scaffold(project.db_path, 2) is None


def test_single_elite_reinvent_retains_automatic_mcs_error(project):
    prepare_iteration_scaffold(project, 1, create=True)
    project.sidechain_library = None
    project.elite_mcs_smarts = None
    project.gbsa_elite_count = 1
    with pytest.raises(RuntimeError, match='At least two input ligands'):
        prepare_iteration_scaffold(project, 2, [(100, 'A', 'CCc1ccc(CC)cc1', 1, -10, -40)], create=True)
    assert database.iteration_scaffold(project.db_path, 2) is None


@pytest.mark.parametrize('fixed', [False, True])
def test_single_elite_respects_supplied_and_pinned_cores(project, monkeypatch, fixed):
    project.elite_mcs_fixed = fixed
    prepare_iteration_scaffold(project, 1, create=True)
    project.gbsa_elite_count = 1
    project.elite_mcs_smarts = None if fixed else '[*:0]Cc1ccc([*:1])cc1'
    monkeypatch.setattr(combichem, 'prepare_single_elite_scaffold',
                        lambda *a: pytest.fail('replaced supplied/pinned scaffold'))
    prepare_iteration_scaffold(project, 2, [(100, 'A', 'CCc1ccc(CC)cc1', 1, -10, -40)], create=True)
    saved = database.iteration_scaffold(project.db_path, 2)
    assert saved['source'] == ('fixed_elite_mcs' if fixed else 'user_elite_mcs')


def test_fixed_combichem_core_stays_pinned_with_fresh_pools_per_iteration(project, monkeypatch):
    project.elite_mcs_fixed = True
    prepare_iteration_scaffold(project, 1, create=True)
    first_core = database.iteration_scaffold(project.db_path, 1)
    seeds = iter([101, 202])
    monkeypatch.setattr(combichem.secrets, 'randbits', lambda _: next(seeds))
    first = combichem.prepare_plan(project, 1, project.output / 'iter1/combichem')
    elites = [(100, 'A', 'CCc1ccc(C)cc1', 1, -10, -40),
              (101, 'B', 'CCCc1ccc(C)cc1', 1, -9, -30)]
    prepare_iteration_scaffold(project, 2, elites, create=True)
    second = combichem.prepare_plan(project, 2, project.output / 'iter2/combichem')
    assert first['scaffold_smiles'] == second['scaffold_smiles'] == first_core['scaffold_smiles']
    assert first['seed'] != second['seed']
    assert first['pools'] != second['pools']


@pytest.mark.parametrize('command', ['run', 'generate'])
def test_combichem_cli_with_no_prior_and_elite_override(project, monkeypatch, command):
    toml = project.output / 'test.toml'
    toml.write_text('''dock-grid = "grid.zip"
reference-pose = "refs.maegz"
out-dir = "."
sample-size = 4
target-count = 4
max-iteration = 5
host = "localhost"
synthetic-feasibility-enabled = false
reference-mcs = "c1ccccc1"
elite-mcs = "[*:0]Cc1ccc([*:1])cc1"
sidechain-library = "edges.smi"
''')
    monkeypatch.setattr(_core, 'initialize_project_database', lambda **k: None)
    monkeypatch.setattr(driver, 'reference_work_pending', lambda args: False)
    monkeypatch.setattr(driver, 'build_stages', lambda *args: [('generate', 'molnova.stages.generate')])
    def worker(stage, cmd, cwd, stop):
        generate.main(cmd[3:])
        return 0
    monkeypatch.setattr(driver, 'stream_process', worker)
    core = '[*:0]c1ccc([*:1])cc1'
    cli.main([command, str(toml), '--iteration', '1', '--elite-mcs', core])
    assert database.iteration_compound_count(project.db_path, 1) == 4
    assert database.iteration_scaffold(project.db_path, 1)['supplied_elite_mcs'] == core
    assert database.combichem_plan(project.db_path, 1)['total_combinations'] == '16'
