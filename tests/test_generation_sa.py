from types import SimpleNamespace

import pytest

from molnova import _core, database
from molnova.chemistry.sa_score import calculate_sa_score, calculate_sa_scores
from molnova.config import sa_score_process_count
from molnova.stages import generate


@pytest.fixture
def project(tmp_path):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
    return SimpleNamespace(db_path=db, output=tmp_path, target_count=2,
                           max_iteration=2, libinvent_prior='prior',
                           libinvent_scaffold_file=tmp_path / 'scaffold.smi',
                           sample_size=10, synthetic_feasibility_enabled=False,
                           sa_score_cpus=1)


def test_generation_stores_real_sa_scores_without_aizynth(project, monkeypatch):
    csv = project.output / 'generated.csv'
    csv.write_text('SMILES\nCCO\nc1ccccc1\n')
    monkeypatch.setattr(_core, 'run_libinvent_sampling', lambda **kw: csv)
    generate._run_stage(project, 1)
    with database.connect(project.db_path) as conn:
        rows = conn.execute('SELECT smiles,sa_score,state,synthetic_feasibility FROM compound').fetchall()
    assert len(rows) == 2
    for smiles, score, state, feasible in rows:
        assert score == pytest.approx(calculate_sa_score(smiles))
        assert state == 'generated'
        assert feasible is None


def test_completed_generation_backfills_only_missing_scores(project, monkeypatch):
    with database.connect(project.db_path) as conn:
        conn.executemany(
            "INSERT INTO compound(name,smiles,iteration,state,sa_score,docking_score,gbsa_score,failed_stage,failure_message) "
            "VALUES (?, ?, 1, ?, ?, -8, -40, 'gbsa', 'diagnostic')",
            [('a', 'CCO', 'gbsa_done', 3.25), ('b', 'CC', 'failed', None)],
        )
        conn.execute("INSERT INTO compound(name,smiles,iteration,state) VALUES ('ref','C',0,'reference')")
        before = conn.execute('SELECT id,state,docking_score,gbsa_score,failed_stage,failure_message FROM compound').fetchall()
    monkeypatch.setattr(_core, 'run_libinvent_sampling', lambda **kw: pytest.fail('resampled'))
    generate._run_stage(project, 1)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT id,state,docking_score,gbsa_score,failed_stage,failure_message FROM compound').fetchall() == before
        assert conn.execute('SELECT sa_score FROM compound ORDER BY id').fetchall() == [
            (3.25,), (pytest.approx(calculate_sa_score('CC')),), (None,)]
    monkeypatch.setattr(generate, 'calculate_sa_scores', lambda *a: pytest.fail('rescored'))
    generate._run_stage(project, 1)


def test_scoring_failure_retains_generation_for_retry(project, monkeypatch):
    csv = project.output / 'generated.csv'
    csv.write_text('SMILES\nCCO\nCC\n')
    monkeypatch.setattr(_core, 'run_libinvent_sampling', lambda **kw: csv)
    scorer = generate.calculate_sa_scores
    def fail(*args):
        raise RuntimeError('scoring failed')
    monkeypatch.setattr(generate, 'calculate_sa_scores', fail)
    with pytest.raises(RuntimeError, match='scoring failed'):
        generate._run_stage(project, 1)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT count(*),count(sa_score) FROM compound').fetchone() == (2, 0)
    monkeypatch.setattr(generate, 'calculate_sa_scores', scorer)
    monkeypatch.setattr(_core, 'run_libinvent_sampling', lambda **kw: pytest.fail('resampled'))
    generate._run_stage(project, 1)
    assert database.missing_sa_scores(project.db_path, 1) == []


def test_local_parallel_scores_match_serial_scores(monkeypatch):
    from molnova.chemistry import sa_score
    monkeypatch.setattr(sa_score.os, 'sched_getaffinity', lambda _: {0, 1}, raising=False)
    compounds = [(3, 'CCO'), (10, 'c1ccccc1'), (20, 'CC(=O)O')]
    assert calculate_sa_scores(compounds, 2) == calculate_sa_scores(compounds, 1)
    assert calculate_sa_scores([], 2) == []


def test_invalid_molecule_never_partially_stores_scores(project):
    with database.connect(project.db_path) as conn:
        conn.executemany("INSERT INTO compound(name,smiles,iteration,state) VALUES (?, ?, 1, 'generated')",
                         [('a', 'CCO'), ('b', 'invalid')])
    with pytest.raises(ValueError, match='invalid or empty'):
        generate.store_iteration_sa_scores(project, 1)
    assert len(database.missing_sa_scores(project.db_path, 1)) == 2


@pytest.mark.parametrize('value', [True, 0, -1, 2.5, '8'])
def test_invalid_cpu_settings(value):
    with pytest.raises(ValueError, match='sa-score-cpus'):
        sa_score_process_count({'sa-score-cpus': value})
    with pytest.raises(ValueError, match='sa-score-cpus'):
        calculate_sa_scores([], value)


def test_cpu_setting_loaded_from_toml(tmp_path):
    toml = tmp_path / 'project.toml'
    content = '''libinvent-prior = "prior"
dock-grid = "grid.zip"
reference-pose = "refs.maegz"
out-dir = "output"
sample-size = 10
target-count = 2
max-iteration = 2
host = "localhost"
synthetic-feasibility-enabled = false
'''
    toml.write_text(content)
    assert _core.load_project_toml(toml)['sa_score_cpus'] == 1
    assert _core.load_project_toml(toml)['sa_accepted_ratio'] == .75
    toml.write_text(content + 'sa-score-cpus = 128\nsa-accepted-ratio = 0.5\n')
    assert _core.load_project_toml(toml)['sa_score_cpus'] == 128
    assert _core.load_project_toml(toml)['sa_accepted_ratio'] == .5
