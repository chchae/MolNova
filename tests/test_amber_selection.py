from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from molnova import _core, amber_gbsa, database
from molnova.stages import amber_gbsa as stage


@pytest.fixture
def population(tmp_path):
    db = tmp_path / 'selection.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany(
            "INSERT INTO compound(id,name,smiles,iteration,state,docking_score) "
            "VALUES (?,?,'CCO',1,'docked',?)",
            [(i, f'GEN_{i}', -i) for i in range(1, 1001)])
    return db


def finish(db, ids):
    database.store_amber_results(db, 1,
        [{'id':i, 'score':-(1001-i)} for i in ids], ids)


@pytest.mark.parametrize('enabled', [False, True])
def test_mode_defaults_select_prime_50_or_amber_500_then_prime_50(population, tmp_path, enabled):
    from molnova.config import amber_settings
    settings = amber_settings({'out-dir': 'output', 'amber_gbsa': enabled}, tmp_path)
    args = SimpleNamespace(**settings)
    keep = database.amber_keep_count(args)
    if enabled:
        ids = database.claim_amber_candidates(population, 1, 1000, keep)
        assert ids == list(range(1000, 500, -1))
        assert database.claim_primg_gbsa_candidates(population, 1, args.primg_gbsa_input_count, 1000, keep) == []
        finish(population, ids)
        expected = list(range(501, 551))
    else:
        expected = list(range(1000, 950, -1))
    assert database.claim_primg_gbsa_candidates(
        population, 1, args.primg_gbsa_input_count, 1000, keep
    ) == expected
    with database.connect(population) as conn:
        assert conn.execute('SELECT COUNT(amber_gbsa_score) FROM compound').fetchone()[0] == (500 if enabled else 0)


def test_only_glide_500_are_claimed_and_amber_200_go_to_prime(population):
    db = population
    assert database.claim_amber_candidates(db, 1, 1001, 500) == []
    ids = database.claim_amber_candidates(db, 1, 1000, 500)
    assert ids == list(range(1000, 500, -1))
    assert database.claim_primg_gbsa_candidates(db, 1, 200, 1000, 500) == []
    finish(db, ids)
    assert database.amber_completion_reason(db, 1, 1000, 500) is None
    assert database.stage_completion_reason(db, 1, 'amber-gbsa', 1000, 200,
                                            amber_keep_count=500) is None
    assert database.claim_amber_candidates(db, 1, 1000, 500) == []
    assert database.claim_primg_gbsa_candidates(db, 1, 200, 1000, 500) == list(range(501, 701))
    with database.connect(db) as conn:
        assert conn.execute('SELECT COUNT(amber_gbsa_score) FROM compound').fetchone()[0] == 500
        assert conn.execute("SELECT COUNT(*) FROM compound WHERE id<=500 AND state='docked'").fetchone()[0] == 500


def test_atomic_claims_never_refill_from_outside_glide_top_500(population):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _:database.claim_amber_candidates(population,1,1000,500), range(2)))
    assert len(set(results[0]) & set(results[1])) == 0
    assert sorted(results[0] + results[1]) == list(range(501,1001))


def test_scores_failures_and_ties_keep_their_glide_positions(population):
    db = population
    with database.connect(db) as conn:
        conn.execute("UPDATE compound SET state='primg_gbsa_done',primg_gbsa_score=-50,amber_gbsa_score=-20 WHERE id=1000")
        conn.execute("UPDATE compound SET state='failed',failed_stage='amber_gbsa' WHERE id=999")
        conn.execute('UPDATE compound SET docking_score=-998 WHERE id IN (997,998)')
    assert database.claim_amber_candidates(db, 1, 1000, 3) == [997]
    finish(db, [997])
    assert database.claim_amber_candidates(db, 1, 1000, 3) == []
    assert database.amber_completion_reason(db, 1, 1000, 3) is None
    assert database.claim_primg_gbsa_candidates(db, 1, 2, 1000, 3) == [997]
    with database.connect(db) as conn:
        assert conn.execute('SELECT state,amber_gbsa_score FROM compound WHERE id=998').fetchone() == ('docked',None)


def test_legacy_outside_scores_are_preserved_but_not_selected(population, tmp_path):
    db = population
    with database.connect(db) as conn:
        conn.execute("UPDATE compound SET amber_gbsa_score=-999,primg_gbsa_score=-100,state='primg_gbsa_done' WHERE id=1")
    ids = database.claim_amber_candidates(db, 1, 1000, 500)
    finish(db, ids)
    assert [r[0] for r in database.prime_selection_rows(db, 1, 200, 500)] == list(range(501,701))
    args = SimpleNamespace(db_path=db, amber_keep_count=500, primg_gbsa_input_count=200)
    amber_gbsa.write_rankings(args, 1, tmp_path)
    assert len((tmp_path/'amber_scores.tsv').read_text().splitlines()) == 502
    assert len((tmp_path/'amber_top.tsv').read_text().splitlines()) == 501
    assert len((tmp_path/'prime_candidates.tsv').read_text().splitlines()) == 201
    assert not any(line.startswith('1\t') for line in (tmp_path/'prime_candidates.tsv').read_text().splitlines()[1:])
    with database.connect(db) as conn:
        assert conn.execute('SELECT amber_gbsa_score,primg_gbsa_score,state FROM compound WHERE id=1').fetchone() == (-999,-100,'primg_gbsa_done')


def test_legacy_outside_running_claim_still_blocks_prime(population):
    db = population
    finish(db, database.claim_amber_candidates(db, 1, 1000, 500))
    with database.connect(db) as conn:
        conn.execute("UPDATE compound SET state='amber_running' WHERE id=1")
    assert 'running Amber' in database.amber_completion_reason(db, 1, 1000, 500)
    assert database.claim_primg_gbsa_candidates(db, 1, 200, 1000, 500) == []
    assert 'running Amber' in database.iteration_completion_reason(db,1,1000,200,1,500)


def test_generation_gate_needs_only_selected_amber_and_prime_results(population):
    db = population
    finish(db, database.claim_amber_candidates(db, 1, 1000, 500))
    assert 'unfinished primg_gbsa' in database.iteration_completion_reason(db,1,1000,200,1,500)
    with database.connect(db) as conn:
        conn.execute("UPDATE compound SET primg_gbsa_score=-40,state='primg_gbsa_done' WHERE id BETWEEN 501 AND 700")
    assert database.iteration_completion_reason(db,1,1000,200,20,500) is None
    with database.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM compound WHERE amber_gbsa_score IS NULL').fetchone()[0] == 500


def test_nonfinite_glide_scores_cannot_enter_amber_selection(population):
    db = population
    with database.connect(db) as conn:
        conn.execute('UPDATE compound SET docking_score=? WHERE id=1000', (float('-inf'),))
        conn.execute('UPDATE compound SET docking_score=? WHERE id=999', (float('inf'),))
        conn.execute('UPDATE compound SET docking_score=NULL WHERE id=998')
    assert database.claim_amber_candidates(db, 1, 1000, 2) == [997,996]


@pytest.mark.parametrize('layout', ['iter1/amber_gbsa', 'iter01/07-amber_gbsa'])
def test_worker_honors_configured_glide_count_and_recovers_legacy_run(population, tmp_path, monkeypatch, layout):
    directory = tmp_path / layout
    directory.mkdir(parents=True)
    args = SimpleNamespace(db_path=population, output=tmp_path, target_count=1000,
        amber_gbsa_enabled=True, amber_keep_count=500, primg_gbsa_input_count=200,
        amber_receptor=tmp_path/'receptor.maegz', amber_charge_method='bcc',
        amber_remote={'host':'t41','remote_work_dir':'/runs'})
    args.amber_receptor.touch()
    def poll(args, iteration, directory, record):
        assert record['compound_ids'] == list(range(1000,500,-1))
        finish(args.db_path, record['compound_ids'])
        (directory/'amber_job.json').unlink()
        return True
    monkeypatch.setattr(amber_gbsa, 'poll', poll)
    stage._run_stage(args, 1)
    assert database.amber_completion_reason(population,1,1000,500) is None
    # Recover a pre-change saved all-pose run intact, even with complete top-500 scores.
    old_ids = database.claim_amber_candidates(population,1,1000,1000)
    assert old_ids == list(range(500,0,-1))
    amber_gbsa.new_record(args,1,directory,old_ids)
    def recover(args, iteration, directory, record):
        assert record['compound_ids'] == old_ids
        finish(args.db_path,old_ids)
        (directory/'amber_job.json').unlink()
        return True
    monkeypatch.setattr(amber_gbsa, 'poll', recover)
    stage._run_stage(args,1)
    assert len(database.amber_rankings(population,1)) == 1000
    assert len(database.amber_rankings(population,1,500)) == 500
