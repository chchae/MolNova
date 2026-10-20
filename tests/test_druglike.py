import csv
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from molnova import _core, database
from molnova.chemistry.descriptors import calculate_rdkit_descriptors
from molnova.chemistry.druglike import filter_druglike
from molnova.stages import generate


@pytest.fixture
def project(tmp_path):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
    database.mark_iteration_workspace_prepared(db, 1)
    with database.connect(db) as conn:
        conn.execute("UPDATE iteration_workspace SET generation_policy='legacy' WHERE iteration=1")
    return SimpleNamespace(db_path=db, output=tmp_path, target_count=2, max_iteration=2)


def test_ro5_gate_qed_sa_ties_stable_order_and_odd_half():
    rows = [
        dict(id='bad', qed=1, sa_score=1, ro5_violations=2),
        dict(id='lower', qed=.7, sa_score=3, ro5_violations=0),
        dict(id='tie1', qed=.8, sa_score=2, ro5_violations=1),
        dict(id='tie2', qed=.8, sa_score=2, ro5_violations=0),
        dict(id='best', qed=.8, sa_score=1, ro5_violations=0),
    ]
    original = [dict(row) for row in rows]
    result = filter_druglike(rows)
    assert [r['id'] for r in result] == [r['id'] for r in rows]
    assert [r['id'] for r in result if r['druglike_selected']] == ['tie1', 'tie2', 'best']
    assert result[0]['druglike_reject_reason'] == 'ro5_violations'
    assert result[1]['druglike_reject_reason'] == 'qed_rank'
    assert all(r['druglike_reject_reason'] is None for r in result[2:])
    assert rows == original
    # Full ties at a cutoff keep the original order.
    tied = [dict(id=i, qed=.5, sa_score=2, ro5_violations=0) for i in range(4)]
    assert [r['id'] for r in filter_druglike(tied) if r['druglike_selected']] == [0, 1]
    assert filter_druglike([]) == []
    assert filter_druglike(tied[:1])[0]['druglike_selected']


def test_low_eligible_count_does_not_relax_ro5():
    rows = [dict(id=i, qed=.9, sa_score=1, ro5_violations=2) for i in range(6)]
    rows[4]['ro5_violations'] = 1
    result = filter_druglike(rows)
    assert [r['id'] for r in result if r['druglike_selected']] == [4]


def test_filter_precedes_db_target_and_retains_audit_and_all_properties(project):
    with database.connect(project.db_path) as conn:
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,docking_score) VALUES ('ref','CCO',0,'reference',-8)")
    sampled = project.output / 'generated.csv'
    sampled.write_text('SMILES\nCCO\nOCC\ninvalid\n\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O\nOc1ccccc1\n' + 'C' * 36 + '\n')
    assert _core.insert_generated(sampled, 1, 2, project.db_path) == 2
    with database.connect(project.db_path) as conn:
        rows = conn.execute('SELECT smiles,rdk_qed,rdk_ro5_violations,sa_score,rdk_sa_score FROM compound WHERE iteration=1 ORDER BY id').fetchall()
        assert [r[0] for r in rows] == ['CC(=O)Oc1ccccc1C(=O)O', 'Oc1ccccc1']
        assert all(r[1] > .5 and r[2] <= 1 and r[3] == r[4] for r in rows)
        assert conn.execute('SELECT state,docking_score FROM compound WHERE iteration=0').fetchone() == ('reference', -8)
    assert database.missing_rdkit_descriptors(project.db_path, 1) == []
    with next(project.output.glob('druglike_selection-*.csv')).open() as stream:
        audit = list(csv.DictReader(stream))
    assert len(audit) == 4  # Existing, internal duplicate and invalid SMILES excluded first.
    assert [r['druglike_selected'] for r in audit] == ['False', 'True', 'True', 'False']
    assert audit[0]['druglike_reject_reason'] == 'qed_rank'
    assert audit[3]['druglike_reject_reason'] == 'ro5_violations'
    assert set(database.RDK_DESCRIPTOR_TYPES) <= set(audit[0])
    # Subsequent target-count capping preserves input order within the passing half.
    another = project.output / 'another.csv'
    another.write_text('SMILES\nCCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O\nOc1ccccc1\n')
    assert _core.insert_generated(another, 2, 1, project.db_path) == 1
    assert len(list(project.output.glob('druglike_selection-*.csv'))) == 2


def test_underfilled_batch_can_be_replenished_without_duplicates(project):
    sampled = project.output / 'generated.csv'
    sampled.write_text('SMILES\nCCO\n' + '\n'.join('C' * n for n in [36, 37, 38]) + '\n')
    assert _core.insert_generated(sampled, 1, 2, project.db_path) == 1
    assert 'generation incomplete' in database.stage_completion_reason(project.db_path, 1, 'generate', 2, 0)
    assert generate.choose_iteration(project, None) == 1
    sampled.write_text('SMILES\nCCO\nc1ccccc1\nCCN\n')
    assert _core.insert_generated(sampled, 1, 2, project.db_path) == 1
    assert database.stage_completion_reason(project.db_path, 1, 'generate', 2, 0) is None
    assert _core.insert_generated(sampled, 1, 2, project.db_path) == 0
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT count(*),count(DISTINCT smiles) FROM compound WHERE iteration=1').fetchone() == (2, 2)


def test_concurrent_inserts_respect_total_iteration_target(project):
    molecules = ['CCO', 'CCN', 'CCC', 'CCCC']
    candidates = [dict(calculate_rdkit_descriptors(s), smiles=s, druglike_selected=True) for s in molecules]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: database.insert_generated_candidates(project.db_path, 1, 2, candidates), range(2)))
    assert sorted(results) == [0, 2]
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT count(*),count(DISTINCT name) FROM compound').fetchone() == (2, 2)


def test_legacy_generation_reuses_precomputed_descriptors(project, monkeypatch):
    project.libinvent_prior = 'prior'
    project.libinvent_scaffold_file = project.output / 'scaffold.smi'
    project.sample_size = 4
    sampled = project.output / 'generated.csv'
    sampled.write_text('SMILES\nCCO\nc1ccccc1\nCC(=O)Oc1ccccc1C(=O)O\nOc1ccccc1\n')
    monkeypatch.setattr(_core, 'run_libinvent_sampling', lambda **kw: sampled)
    original = generate.calculate_rdkit_descriptors_batch
    calls = []
    def calculate(compounds, cpus):
        calls.append(list(compounds))
        assert len(calls) == 1, 'properties recalculated after insertion'
        return original(calls[-1], cpus)
    monkeypatch.setattr(generate, 'calculate_rdkit_descriptors_batch', calculate)
    generate._run_stage(project, 1)
    assert len(calls[0]) == 4
    assert database.missing_rdkit_descriptors(project.db_path, 1) == []
