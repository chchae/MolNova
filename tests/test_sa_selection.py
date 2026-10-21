from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from molnova import _core, database
from molnova.config import sa_accepted_ratio
from molnova.stages import generate


@pytest.fixture
def project(tmp_path):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        # Two ties at the boundary: lower ID must win regardless of insert order/SMILES.
        conn.executemany(
            "INSERT INTO compound(name,smiles,iteration,state,sa_score) VALUES (?, 'CCO', 1, 'generated', ?)",
            [('high', 8), ('tie-first', 3), ('low', 1), ('tie-last', 3)],
        )
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('ref','C',0,'reference')")
    return SimpleNamespace(db_path=db, target_count=4, sa_accepted_ratio=0.75)


def test_final_selection_and_iteration_gates(project):
    assert database.apply_sa_filter(project.db_path, 1, 4, .75) == (.75, 4, 3)
    assert [row[1] for row in database.ligprep_pending_rows(project.db_path, 1, False)] == [
        'tie-first', 'low', 'tie-last']
    with database.connect(project.db_path) as conn:
        assert conn.execute("SELECT state,sa_score FROM compound WHERE name='high'").fetchone() == ('sa_rejected', 8)
        conn.execute("UPDATE compound SET state='docked',docking_score=-8,synthetic_feasibility=1 WHERE state='generated'")
    assert database.stage_completion_reason(project.db_path, 1, 'synthetic', 4, 3, True) is None
    assert database.stage_completion_reason(project.db_path, 1, 'ligprep', 4, 3, True) is None
    assert database.upstream_completion_reason(project.db_path, 1, 4, 'glide', True) is None
    assert database.docking_completion_reason(project.db_path, 1, 4) is None
    assert len(database.claim_primg_gbsa_candidates(project.db_path, 1, 3, 4)) == 3


def test_boundary_ties_rounding_and_reference_exemption(project):
    assert database.apply_sa_filter(project.db_path, 1, 4, .26) == (.26, 4, 2)
    assert [row[1] for row in database.ligprep_pending_rows(project.db_path, 1, False)] == ['tie-first', 'low']
    assert database.apply_sa_filter(project.db_path, 0, 1, .75) is None
    with database.connect(project.db_path) as conn:
        assert conn.execute("SELECT state FROM compound WHERE iteration=0").fetchone() == ('reference',)


def test_selection_is_frozen_on_restart_and_config_change(project):
    original = database.apply_sa_filter(project.db_path, 1, 4, .75)
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state='ligprep_running' WHERE name='low'")
        before = conn.execute('SELECT * FROM compound').fetchall()
    assert database.apply_sa_filter(project.db_path, 1, 4, 1) == original
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT * FROM compound').fetchall() == before


def test_underfilled_iteration_is_not_filtered_early(project):
    assert database.apply_sa_filter(project.db_path, 1, 5, .75) is None
    assert len(database.ligprep_pending_rows(project.db_path, 1, False)) == 4


@pytest.mark.parametrize('score', [None, float('inf'), -1, 11])
def test_invalid_scores_block_selection_atomically(project, score):
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET sa_score=? WHERE name='low'", (score,))
    with pytest.raises(RuntimeError, match='all valid SA scores'):
        database.apply_sa_filter(project.db_path, 1, 4, .75)
    assert len(database.ligprep_pending_rows(project.db_path, 1, False)) == 4


@pytest.mark.parametrize('state', ['ligprep_running', 'synthetic_running', 'docked', 'primg_gbsa_done', 'failed'])
def test_existing_downstream_work_and_results_are_preserved(project, state):
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state=?,failed_stage='glide',failure_message='old' WHERE name='high'", (state,))
        before = conn.execute('SELECT * FROM compound').fetchall()
    assert database.apply_sa_filter(project.db_path, 1, 4, .75) is None
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT * FROM compound').fetchall() == before


def test_concurrent_filters_use_one_selection(project):
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: database.apply_sa_filter(project.db_path, 1, 4, .75), range(2)))
    assert results == [(.75, 4, 3)] * 2
    assert len(database.ligprep_pending_rows(project.db_path, 1, False)) == 3


def test_generation_backfill_applies_filter_without_resampling(project, monkeypatch):
    project.max_iteration = 1
    monkeypatch.setattr(_core, 'run_libinvent_sampling', lambda **kw: pytest.fail('resampled'))
    generate._run_stage(project, 1)
    assert len(database.ligprep_pending_rows(project.db_path, 1, False)) == 3


@pytest.mark.parametrize('value', [True, False, 0, -.1, 1.1, '0.75', float('nan'), float('inf')])
def test_invalid_ratio(value):
    with pytest.raises(ValueError, match='sa-accepted-ratio'):
        sa_accepted_ratio({'sa-accepted-ratio': value})


def test_ratio_configuration():
    assert sa_accepted_ratio({}) == 1.0
    assert sa_accepted_ratio({'sa-accepted-ratio': .5}) == .5
    assert sa_accepted_ratio({'sa-accepted-ratio': 1}) == 1


def test_new_generation_scores_full_population_then_filters(tmp_path, monkeypatch):
    from molnova.chemistry.sa_score import calculate_sa_score
    db = tmp_path / 'new.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
    csv = tmp_path / 'generated.csv'
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'C1CC2CCC1C2',
              'CCN', 'c1ccncc1', 'Oc1ccccc1', 'CCOc1ccccc1']
    csv.write_text('SMILES\n' + '\n'.join(smiles) + '\n')
    args = SimpleNamespace(db_path=db, output=tmp_path, target_count=8,
                           max_iteration=1, libinvent_prior='prior',
                           libinvent_scaffold_file=tmp_path / 'scaffold.smi',
                           sample_size=10, sa_score_cpus=1, sa_accepted_ratio=.75)
    monkeypatch.setattr(_core, 'run_libinvent_sampling', lambda **kw: csv)
    generate._run_stage(args, 1)
    from molnova.stages import druglike
    druglike._run_stage(args, 1)
    with database.connect(db) as conn:
        rows = conn.execute('SELECT smiles,sa_score,state FROM compound ORDER BY sa_score,id').fetchall()
    assert len(rows) == 8
    assert sum(state == 'generated' for _, _, state in rows) == 3
    assert sum(state == 'druglike_rejected' for _, _, state in rows) == 5
    for smiles, score, _ in rows:
        assert score == pytest.approx(calculate_sa_score(smiles))


def test_synthetic_search_receives_only_accepted_compounds(tmp_path, monkeypatch):
    import gzip
    import json
    from molnova.stages import synthetic_feasibility as synthetic
    db = tmp_path / 'synthetic.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany(
            "INSERT INTO compound(name,smiles,iteration,state,sa_score) VALUES (?, ?, 1, 'generated', ?)",
            [('a', 'CC', 1), ('b', 'CCC', 2), ('c', 'CCCC', 3), ('excluded', 'C', 8)],
        )
    config = tmp_path / 'aizynth.yml'
    config.write_text('search: {}\n')
    args = SimpleNamespace(db_path=db, output=tmp_path, target_count=4,
                           sa_accepted_ratio=.75, synthetic_feasibility_enabled=True,
                           aizynth_config=config, aizynth_cli='aizynthcli')
    monkeypatch.setattr(synthetic.shutil, 'which', lambda _: '/bin/aizynthcli')
    def run(command, **kwargs):
        from pathlib import Path
        targets = Path(command[command.index('--smiles') + 1]).read_text().splitlines()
        assert targets == ['CC', 'CCC', 'CCCC']
        with gzip.open(command[command.index('--output') + 1], 'wt') as stream:
            json.dump({'data': [{'target': smiles, 'number_of_solved_routes': 1} for smiles in targets]}, stream)
    monkeypatch.setattr(synthetic.subprocess, 'run', run)
    assert synthetic.process_iteration(args, 1) == 3
    assert database.stage_completion_reason(db, 1, 'synthetic', 4, 3, True) is None
    assert len(database.ligprep_pending_rows(db, 1, True)) == 3
