import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, driver
from molnova.chemistry.druglike import filter_druglike
from molnova.stages import druglike, generate, fep


@pytest.fixture
def raw_project(tmp_path):
    db = tmp_path / 'test.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
    return SimpleNamespace(db_path=db, output=tmp_path, target_count=4, max_iteration=2,
                           sample_size=4, libinvent_prior=Path('prior'),
                           libinvent_scaffold_file=tmp_path/'scaffold.smi',
                           sa_score_cpus=1, sa_accepted_ratio=1.0,
                           synthetic_feasibility_enabled=False)


def generate_raw(args, monkeypatch):
    path = args.output / 'generated.csv'
    path.write_text('SMILES\nCCO\nCCN\nc1ccccc1\nCC(=O)O\n')
    monkeypatch.setattr(_core, 'run_libinvent_sampling', lambda **kw: path)
    generate._run_stage(args, 1)


def test_generation_commits_all_before_iteration_wide_druglike(raw_project, monkeypatch):
    args = raw_project
    generate_raw(args, monkeypatch)
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT COUNT(*),COUNT(rdk_qed) FROM compound').fetchone() == (4, 0)
    assert database.upstream_completion_reason(args.db_path, 1, 4, 'ligprep')
    druglike._run_stage(args, 1)
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT COUNT(*),COUNT(rdk_qed) FROM compound').fetchone() == (4, 4)
        assert conn.execute("SELECT COUNT(*) FROM compound WHERE state='druglike_rejected'").fetchone()[0] == 2
    expected = filter_druglike(database.druglike_population(args.db_path, 1))
    pending = database.ligprep_pending_rows(args.db_path, 1, False)
    assert {row[0] for row in pending} == {row['id'] for row in expected if row['druglike_selected']}
    assert database.upstream_completion_reason(args.db_path, 1, 4, 'ligprep') is None
    assert len(list(csv.DictReader((args.output/'iter01/03-druglike/druglike_selection.csv').open()))) == 4
    # Both decisions and reporting survive configuration changes and a lost report.
    saved = database.druglike_report(args.db_path, 1)
    args.sa_accepted_ratio = .1
    (args.output/'iter01/03-druglike/druglike_selection.csv').unlink()
    druglike._run_stage(args, 1)
    assert database.druglike_report(args.db_path, 1) == saved


def test_property_failure_preserves_raw_population_and_blocks_downstream(raw_project, monkeypatch):
    generate_raw(raw_project, monkeypatch)
    original = generate.calculate_rdkit_descriptors_batch
    monkeypatch.setattr(generate, 'calculate_rdkit_descriptors_batch',
                        lambda *a: (_ for _ in ()).throw(RuntimeError('descriptor failure')))
    with pytest.raises(RuntimeError, match='descriptor failure'):
        druglike._run_stage(raw_project, 1)
    with database.connect(raw_project.db_path) as conn:
        assert conn.execute('SELECT COUNT(*),COUNT(rdk_qed) FROM compound').fetchone() == (4, 0)
    assert database.druglike_completion_reason(raw_project.db_path, 1)
    monkeypatch.setattr(generate, 'calculate_rdkit_descriptors_batch', original)
    druglike._run_stage(raw_project, 1)
    assert database.druglike_completion_reason(raw_project.db_path, 1) is None


def test_druglike_commit_is_atomic_and_refuses_changed_population(raw_project, monkeypatch):
    args = raw_project
    generate_raw(args, monkeypatch)
    generate.store_iteration_descriptors(args, 1)
    selected = filter_druglike(database.druglike_population(args.db_path, 1))
    with pytest.raises(RuntimeError, match='population changed'):
        database.apply_druglike_selection(args.db_path, 1, 4, selected[:-1])
    assert database.druglike_completion_reason(args.db_path, 1)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: database.apply_druglike_selection(args.db_path, 1, 4, selected), range(2)))
    assert results == [(4, 2, 1.0), (4, 2, 1.0)]
    assert len(database.druglike_report(args.db_path, 1)) == 4


def test_old_workspace_policy_is_not_silently_converted(raw_project):
    args = raw_project
    with database.connect(args.db_path) as conn:
        conn.execute('CREATE TABLE iteration_workspace(iteration INTEGER PRIMARY KEY,prepared_at TEXT)')
        conn.execute("INSERT INTO iteration_workspace VALUES (1,'old')")
    database.mark_iteration_workspace_prepared(args.db_path, 2)
    assert not database.uses_raw_generation(args.db_path, 1)
    assert database.uses_raw_generation(args.db_path, 2)


@pytest.fixture
def ranked_project(raw_project):
    args = raw_project
    args.target_count = 6
    args.gbsa_input_count = 2
    args.fep_input_count = 2
    args.amber_gbsa_enabled = False
    with database.connect(args.db_path) as conn:
        conn.executemany("INSERT INTO compound(id,name,smiles,iteration,state,docking_score) "
                         "VALUES (?,?,'CCO',1,'docked',?)",
                         [(i, f'C{i}', -i) for i in range(1, 7)])
    return args


def test_prime_falls_back_only_after_amber_finishes(ranked_project):
    args = ranked_project
    claimed = database.claim_amber_candidates(args.db_path, 1, 6, 4)
    assert claimed == [6, 5, 4, 3]
    assert database.claim_mmgbsa_candidates(args.db_path, 1, 2, 6, 4) == []
    database.store_amber_results(args.db_path, 1,
        [{'id': cid, 'error': 'parameterization failed'} for cid in claimed], claimed)
    assert database.iteration_completion_reason(args.db_path, 1, 6, 2, 0, 4)
    assert database.claim_mmgbsa_candidates(args.db_path, 1, 2, 6, 4) == [6, 5]


def test_partial_amber_results_are_not_filled_with_docking(ranked_project):
    args = ranked_project
    claimed = database.claim_amber_candidates(args.db_path, 1, 6, 4)
    database.store_amber_results(args.db_path, 1,
        [{'id': cid, **({'score': -30} if cid == 3 else {'error': 'failed'})} for cid in claimed], claimed)
    assert database.claim_mmgbsa_candidates(args.db_path, 1, 2, 6, 4) == [3]


@pytest.mark.parametrize('source,column', [('prime','gbsa_score'),('amber','amber_gbsa_score'),('docking','docking_score')])
def test_fep_source_fallback_freeze_and_atomic_import(ranked_project, source, column):
    args = ranked_project
    with database.connect(args.db_path) as conn:
        # Completed, permanent Prime failures allow lower-tier FEP fallback.
        conn.execute("UPDATE compound SET state='failed',failed_stage='gbsa'")
        if source != 'docking':
            conn.execute(f'UPDATE compound SET {column}=-40 WHERE id=1')
        if source == 'prime':
            conn.execute("UPDATE compound SET state='gbsa_done' WHERE id=1")
    output = fep.export_candidates(args, 1)
    rows = list(csv.DictReader(output.open(), delimiter='\t'))
    assert {r['ranking_source'] for r in rows} == {source}
    expected = [6, 5] if source == 'docking' else [1]
    assert [int(r['id']) for r in rows] == expected
    with database.connect(args.db_path) as conn:
        conn.execute('UPDATE compound SET gbsa_score=-999 WHERE id=2')
    args.fep_input_count = 5
    assert [r['id'] for r in database.fep_candidates(args.db_path, 1, 5)] == expected
    scores = args.output/'fep.tsv'
    scores.write_text(f'id\tfep_score\n{expected[0]}\t-10\n2\t-20\n')
    with pytest.raises(ValueError, match='outside'):
        fep.import_scores(args, scores, 1)
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT COUNT(fep_score) FROM compound').fetchone()[0] == 0
    scores.write_text(f'id\tfep_score\n{expected[0]}\t-10\n')
    assert fep.import_scores(args, scores, 1) == 1


def test_fep_waits_for_pending_prime_and_ignores_nonfinite_scores(ranked_project):
    args = ranked_project
    with pytest.raises(RuntimeError, match='unfinished Prime'):
        fep.export_candidates(args, 1)
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='failed',failed_stage='gbsa',gbsa_score=?", (float('inf'),))
    rows = database.fep_candidates(args.db_path, 1, 2, target_count=6, prime_count=2)
    assert [row['id'] for row in rows] == [6,5]
    assert {row['ranking_source'] for row in rows} == {'docking'}


def test_driver_stage_order():
    assert [stage for stage, _ in driver.build_stages(False, True, True)] == [
        'generate','druglike','ligprep','glide','amber-gbsa','mmgbsa','fep']


def test_amber_failure_history_survives_prime_retry_and_completion(ranked_project):
    args = ranked_project
    claimed = database.claim_amber_candidates(args.db_path, 1, 6, 4)
    database.store_amber_results(args.db_path, 1,
        [{'id': cid, 'error': 'parameterization failed'} for cid in claimed], claimed)
    assert database.claim_mmgbsa_candidates(args.db_path, 1, 2, 6, 4) == [6,5]
    assert database.amber_completion_reason(args.db_path, 1, 6, 4) is None
    with database.connect(args.db_path) as conn:
        # A retryable Prime failure replaces the current failure field.
        conn.execute("UPDATE compound SET state='docked',failed_stage='gbsa',failure_message='retry' WHERE id IN (5,6)")
    assert database.amber_completion_reason(args.db_path, 1, 6, 4) is None
    assert database.claim_amber_candidates(args.db_path, 1, 6, 4) == []
    assert database.claim_mmgbsa_candidates(args.db_path, 1, 2, 6, 4) == [6,5]
    score_file = args.output/'prime.tsv'
    score_file.write_text('id\tgbsa_score\n6\t-50\n5\t-40\n')
    _core.update_gbsa_scores(score_file, args.db_path, [6,5])
    assert database.iteration_completion_reason(args.db_path, 1, 6, 2, 1, 4) is None
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM amber_failure').fetchone()[0] == 4


def test_druglike_results_do_not_depend_on_generation_batch_size(raw_project):
    from molnova.chemistry.descriptors import calculate_rdkit_descriptors
    molecules = ['CCO','CCN','c1ccccc1','CC(=O)O']
    first = raw_project.db_path
    second = raw_project.output/'second.sqlite'
    for path in [first, second]:
        with database.connect(path) as conn:
            _core.create_sqlite_schema(conn)
        database.mark_iteration_workspace_prepared(path, 1)
    database.insert_generated_candidates(first, 1, 4, [dict(smiles=m) for m in molecules], raw=True)
    for start in [0,2]:
        database.insert_generated_candidates(second, 1, 4, [dict(smiles=m) for m in molecules[start:start+2]], raw=True)
    for path in [first, second]:
        population = database.druglike_population(path, 1)
        database.store_rdkit_descriptors(path, [(r['id'], calculate_rdkit_descriptors(r['smiles'])) for r in population])
        selected = filter_druglike(database.druglike_population(path, 1))
        database.apply_druglike_selection(path, 1, 4, selected)
    assert database.druglike_report(first, 1) == database.druglike_report(second, 1)
    with pytest.raises(RuntimeError, match='frozen'):
        database.insert_generated_candidates(first, 1, 5, [dict(smiles='CCCl')], raw=True)


def test_saved_uncertain_job_blocks_fep_before_any_snapshot(ranked_project):
    import json
    args = ranked_project
    root = args.output/'iter01/08-mmgbsa'
    root.mkdir(parents=True)
    (root/'prime_job.json').write_text(json.dumps(dict(job_id=None, launching=True)))
    with pytest.raises(RuntimeError, match='saved external'):
        fep._run_stage(args, SimpleNamespace(import_scores=None, iteration=1))
    with database.connect(args.db_path) as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='iteration_fep_selection'").fetchone()


def test_import_without_iteration_ignores_other_unexported_raw_iteration(ranked_project):
    args = ranked_project
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='gbsa_done',gbsa_score=-id")
    fep.export_candidates(args, 1)
    database.mark_iteration_workspace_prepared(args.db_path, 2)
    with database.connect(args.db_path) as conn:
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,docking_score) VALUES ('later','CCN',2,'docked',-20)")
    score_file = args.output/'fep.tsv'
    score_file.write_text('id\tfep_score\n6\t-50\n')
    assert fep.import_scores(args, score_file) == 1


def test_fep_import_rejects_conflicting_scores_without_partial_update(ranked_project):
    args = ranked_project
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='gbsa_done',gbsa_score=-id")
    fep.export_candidates(args, 1)
    with database.connect(args.db_path) as conn:
        conn.execute('UPDATE compound SET fep_score=-10 WHERE id=6')
    score_file = args.output/'fep.tsv'
    score_file.write_text('id\tfep_score\n5\t-30\n6\t-20\n')
    with pytest.raises(ValueError, match='Conflicting'):
        fep.import_scores(args, score_file, 1)
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT fep_score FROM compound WHERE id=5').fetchone() == (None,)
        assert conn.execute('SELECT fep_score FROM compound WHERE id=6').fetchone() == (-10,)
