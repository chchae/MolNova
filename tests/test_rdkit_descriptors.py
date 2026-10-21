from types import SimpleNamespace

import pytest

from molnova import _core, database
from molnova.chemistry import descriptors
from molnova.stages import generate


@pytest.fixture
def project(tmp_path):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
    return SimpleNamespace(db_path=db, output=tmp_path, target_count=2, max_iteration=1,
                           sa_score_cpus=1, synthetic_feasibility_enabled=False)


def test_ethanol_all_requested_properties():
    values = descriptors.calculate_rdkit_descriptors('CCO')
    assert set(values) == set(database.RDK_DESCRIPTOR_TYPES)
    assert values == pytest.approx({
        'rdk_sa_score': 1.9802570386349831, 'rdk_qed': .40680796565539457,
        'rdk_mw': 46.069, 'rdk_clogp': -.0014, 'rdk_tpsa': 20.23,
        'rdk_hbd': 1, 'rdk_hba': 1, 'rdk_rotatable_bonds': 0,
        'rdk_formal_charge': 0, 'rdk_heavy_atoms': 3, 'rdk_aromatic_rings': 0,
        'rdk_fraction_csp3': 1, 'rdk_ro5_violations': 0, 'rdk_veber_pass': 1,
        'rdk_pains_alerts': 0, 'rdk_brenk_alerts': 0,
    })


def test_charge_alerts_and_failed_rule_counts_are_descriptors_only():
    charged = descriptors.calculate_rdkit_descriptors('CC[NH+](CC)CC')
    assert charged['rdk_formal_charge'] == 1
    assert charged['rdk_hba'] == 0
    assert charged['rdk_hbd'] == 1
    catechol = descriptors.calculate_rdkit_descriptors('c1ccc(O)c(O)c1')
    assert catechol['rdk_pains_alerts'] >= 1
    assert catechol['rdk_brenk_alerts'] >= 1
    assert catechol['rdk_aromatic_rings'] == 1
    assert catechol['rdk_fraction_csp3'] == 0
    long_chain = descriptors.calculate_rdkit_descriptors('C' * 36)
    assert long_chain['rdk_ro5_violations'] == 2  # MW and cLogP, not HBD/HBA.
    assert long_chain['rdk_veber_pass'] == 0  # >10 strict rotatable bonds.
    polar = descriptors.calculate_rdkit_descriptors('OCC(O)C(O)C(O)C(O)C(O)CO')
    assert polar['rdk_tpsa'] > 140
    assert polar['rdk_rotatable_bonds'] <= 10
    assert polar['rdk_veber_pass'] == 0


def test_migration_preserves_legacy_results_claims_and_scores(tmp_path):
    db = tmp_path / 'legacy.sqlite'
    with database.connect(db) as conn:
        conn.execute('CREATE TABLE compound(id INTEGER PRIMARY KEY, name TEXT, smiles TEXT, '
                     'iteration INTEGER, state TEXT, sa_score REAL, docking_score REAL, prime_gbsa_score REAL)')
        conn.execute("INSERT INTO compound VALUES (1,'a','CCO',1,'prime_gbsa_running',3.25,-8,NULL)")
        original = conn.execute('SELECT * FROM compound').fetchone()
        _core.ensure_schema_columns(conn)
        assert conn.execute('SELECT id,name,smiles,iteration,state,sa_score,docking_score,prime_gbsa_score FROM compound').fetchone() == original
        assert conn.execute('SELECT rdk_sa_score,rdk_qed FROM compound').fetchone() == (3.25, None)
        conn.execute('UPDATE compound SET rdk_qed=.1')
        _core.ensure_schema_columns(conn)
        assert conn.execute('SELECT rdk_sa_score,rdk_qed,state FROM compound').fetchone() == (3.25, .1, 'prime_gbsa_running')


def test_completed_generation_backfills_properties_preserving_every_other_result(project, monkeypatch):
    with database.connect(project.db_path) as conn:
        conn.executemany(
            "INSERT INTO compound(name,smiles,iteration,state,sa_score,docking_score,prime_gbsa_score,failed_stage,failure_message) "
            "VALUES (?, ?, 1, ?, ?, -8,-40,'prime_gbsa','keep')",
            [('a', 'CCO', 'prime_gbsa_done', 3.25), ('b', 'c1ccc(O)c(O)c1', 'failed', None)],
        )
        conn.execute("UPDATE compound SET rdk_qed=.123 WHERE name='a'")
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('ref','C',0,'reference')")
        original = conn.execute('SELECT id,state,docking_score,prime_gbsa_score,failed_stage,failure_message FROM compound').fetchall()
    monkeypatch.setattr(_core, 'run_libinvent_sampling', lambda **kw: pytest.fail('resampled'))
    generate._run_stage(project, 1)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT id,state,docking_score,prime_gbsa_score,failed_stage,failure_message FROM compound').fetchall() == original
        assert conn.execute("SELECT sa_score,rdk_sa_score,rdk_qed FROM compound WHERE name='a'").fetchone() == (3.25, 3.25, .123)
        assert conn.execute("SELECT rdk_mw,rdk_pains_alerts,rdk_brenk_alerts FROM compound WHERE name='b'").fetchone() == (pytest.approx(110.112), 1, 1)
        assert conn.execute('SELECT rdk_mw FROM compound WHERE iteration=0').fetchone() == (None,)
    assert database.missing_rdkit_descriptors(project.db_path, 1) == []
    monkeypatch.setattr(generate, 'calculate_rdkit_descriptors_batch', lambda *a: pytest.fail('recalculated'))
    generate._run_stage(project, 1)


def test_parallel_and_serial_calculation_match(monkeypatch):
    monkeypatch.setattr(descriptors.os, 'sched_getaffinity', lambda _: {0, 1}, raising=False)
    compounds = [(1, 'CCO', None), (8, 'c1ccc(O)c(O)c1', 2.5)]
    assert descriptors.calculate_rdkit_descriptors_batch(compounds, 2) == descriptors.calculate_rdkit_descriptors_batch(compounds, 1)
    assert descriptors.calculate_rdkit_descriptors_batch([], 1) == []


def test_existing_references_backfill_without_filtering_or_result_changes(project, monkeypatch):
    with database.connect(project.db_path) as conn:
        conn.executemany(
            'INSERT INTO compound(name,smiles,iteration,state,sa_score,rdk_qed,'
            'dG_exp,docking_score,prime_gbsa_score,failed_stage,failure_message) '
            'VALUES (?, ?, 0, ?, ?, ?, -10,-8,-40,?,?)',
            [('a', 'CCO', 'prime_gbsa_running', 3.25, .123, None, None),
             ('b', 'C' * 36, 'failed', None, None, 'prime_gbsa', 'keep'),
             ('c', 'c1ccc(O)c(O)c1', 'prime_gbsa_done', None, None, None, None)],
        )
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('gen','C',1,'generated')")
        original = conn.execute('SELECT id,name,smiles,iteration,state,dG_exp,docking_score,'
                                'prime_gbsa_score,failed_stage,failure_message FROM compound ORDER BY id').fetchall()
    # Existing DB must work without reopening the original MAEGZ or Schrödinger.
    _core.initialize_project_database(project.db_path, project.output / 'missing.maegz',
                                      project.output / 'missing-suite')
    assert database.missing_rdkit_descriptors(project.db_path, 0) == []
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT id,name,smiles,iteration,state,dG_exp,docking_score,'
                            'prime_gbsa_score,failed_stage,failure_message FROM compound ORDER BY id').fetchall() == original
        assert conn.execute("SELECT rdk_sa_score,sa_score,rdk_qed FROM compound WHERE name='a'").fetchone() == (3.25, 3.25, .123)
        assert conn.execute("SELECT rdk_ro5_violations FROM compound WHERE name='b'").fetchone() == (2,)
        assert conn.execute('SELECT rdk_mw FROM compound WHERE iteration=1').fetchone() == (None,)
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='iteration_sa_selection'").fetchone() == (0,)
    monkeypatch.setattr(descriptors, 'calculate_rdkit_descriptors_batch', lambda *a: pytest.fail('recalculated'))
    _core.initialize_project_database(project.db_path, project.output / 'missing.maegz',
                                      project.output / 'missing-suite')


def test_reference_property_failure_preserves_rows_for_retry(project):
    with database.connect(project.db_path) as conn:
        conn.executemany("INSERT INTO compound(name,smiles,iteration,state) VALUES (?, ?, 0, 'reference')",
                         [('a', 'CCO'), ('b', 'invalid')])
    with pytest.raises(ValueError, match='invalid or empty'):
        _core.initialize_project_database(project.db_path, project.output / 'missing.maegz',
                                          project.output / 'missing-suite')
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT count(*),count(rdk_mw),count(sa_score) FROM compound').fetchone() == (2, 0, 0)
        conn.execute("UPDATE compound SET smiles='CC' WHERE name='b'")
    _core.initialize_project_database(project.db_path, project.output / 'missing.maegz',
                                      project.output / 'missing-suite')
    assert database.missing_rdkit_descriptors(project.db_path, 0) == []


def test_invalid_molecule_retains_generated_rows_without_partial_properties(project):
    with database.connect(project.db_path) as conn:
        conn.executemany("INSERT INTO compound(name,smiles,iteration,state) VALUES (?, ?, 1, 'generated')",
                         [('a', 'CCO'), ('b', 'invalid')])
    with pytest.raises(ValueError, match='invalid or empty'):
        generate.store_iteration_descriptors(project, 1)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT count(rdk_sa_score),count(rdk_mw),count(sa_score) FROM compound').fetchone() == (0, 0, 0)
    assert len(database.missing_rdkit_descriptors(project.db_path, 1)) == 2


def test_invalid_result_rolls_back_whole_database_batch(project):
    with database.connect(project.db_path) as conn:
        conn.executemany("INSERT INTO compound(name,smiles,iteration,state) VALUES (?, 'CCO', 1, 'generated')", [('a',), ('b',)])
    good = descriptors.calculate_rdkit_descriptors('CCO')
    bad = dict(good, rdk_qed=float('nan'))
    with pytest.raises(ValueError, match='nonfinite'):
        database.store_rdkit_descriptors(project.db_path, [(1, good), (2, bad)])
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT count(rdk_mw) FROM compound').fetchone() == (0,)


def test_sa_filter_prefers_rdk_score_and_accepts_new_prefix_only(project):
    with database.connect(project.db_path) as conn:
        conn.executemany("INSERT INTO compound(name,smiles,iteration,state,rdk_sa_score,sa_score) "
                         "VALUES (?, 'CCO', 1, 'generated', ?, ?)", [('a', 1, 8), ('b', 8, 1)])
    database.apply_sa_filter(project.db_path, 1, 2, .5)
    assert [r[1] for r in database.ligprep_pending_rows(project.db_path, 1, False)] == ['a']


def test_synthetic_import_and_sa_writer_keep_prefix_alias_consistent(project):
    with database.connect(project.db_path) as conn:
        conn.executemany("INSERT INTO compound(name,smiles,iteration,state) VALUES (?, 'CCO', 1, 'generated')", [('a',), ('b',)])
    database.store_sa_scores(project.db_path, [(1, 2.5)])
    database.import_synthetic_results(project.db_path, 1, [(2, 1)], [(2, 3.5)])
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT sa_score,rdk_sa_score FROM compound ORDER BY id').fetchall() == [(2.5, 2.5), (3.5, 3.5)]
