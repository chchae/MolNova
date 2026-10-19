import csv
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
    assert calls == [(0, 4)]
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
