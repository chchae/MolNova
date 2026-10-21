import json
import sqlite3
from types import SimpleNamespace

import pytest

from molnova import _core, cli, config, database, distributions as reports, driver
from molnova.distribution_html import render


@pytest.fixture
def population(tmp_path):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
        conn.executemany('INSERT INTO compound(id,name,smiles,iteration,state,rdk_mw,docking_score,primg_gbsa_score) '
                         'VALUES (?,?,?,?,?,?,?,?)', [
            (1, 'reference', 'CC', 0, 'reference', 30., None, -10.),
            (2, 'first', 'COc1ccc(Cl)cc1', 1, 'primg_gbsa_done', 142., -8., -25.),
            (3, 'second', 'CCOc1ccc(Cl)cc1', 1, 'docked', 156., -7., None),
            (4, 'missing', 'NCc1ccc(Br)cc1', 1, 'generated', None, None, None),
            (5, 'excluded', 'CCN', 1, 'druglike_rejected', 45., None, None),
            (6, 'future', 'NCCCl', 2, 'primg_gbsa_done', 80., -9., -30.),
        ])
        conn.execute('CREATE TABLE compound_druglike_selection(compound_id INTEGER PRIMARY KEY,passed INTEGER,reason TEXT)')
        conn.executemany('INSERT INTO compound_druglike_selection VALUES (?,?,NULL)', [(2, 1), (3, 1), (5, 0)])
    for iteration in (1, 2):
        database.iteration_scaffold(db, iteration, {'scaffold_smiles': '[*:0]c1ccc([*:1])cc1'})
    return SimpleNamespace(db_path=db, output=tmp_path, project='project', primg_gbsa_elite_count=1)


def read_report(args, iteration=1):
    reports.write_report(args, iteration)
    return json.loads((args.output / 'analysis/summary.json').read_text())


def test_report_is_cumulative_read_only_and_preserves_missing_values(population):
    with sqlite3.connect(population.db_path) as conn:
        before = list(conn.iterdump())
    result = read_report(population)
    assert result['population_n'] == 5
    assert [x['iteration'] for x in result['iterations']] == [0, 1]
    current = result['iterations'][1]
    assert current['n'] == 4 and current['selection_unknown'] == 1
    assert current['cohorts']['selected']['n'] == 2
    assert current['cohorts']['primg_gbsa_scored']['n'] == 1
    stats = current['cohorts']['all']['metrics']['docking_score']
    assert stats['n'] == 2 and stats['missing'] == 2
    assert stats['median'] == -7.5 and sum(stats['hist']) == 2
    assert current['cohorts']['all']['metrics']['rdk_qed']['median'] is None
    assert current['diversity']['previous_elite_n'] == 1
    assert current['substituents']['matched_n'] == 3
    assert len(current['substituents']['sites']) == 2
    assert (population.output / 'analysis/iter01.html').is_file()
    for point in result['points']:
        assert (population.output / 'analysis' / point['image']).is_file()
    with sqlite3.connect(population.db_path) as conn:
        assert list(conn.iterdump()) == before


def test_pca_axes_and_existing_coordinates_survive_new_iterations(population):
    first = read_report(population, 1)
    path = population.output / 'analysis/chemical_space_pca.json'
    model = path.read_bytes()
    second = read_report(population, 2)
    assert path.read_bytes() == model
    original = {point['id']: (point['x'], point['y']) for point in first['points']}
    for point in second['points']:
        if point['id'] in original:
            assert (point['x'], point['y']) == pytest.approx(original[point['id']], abs=1e-6)
    assert second['projection']['trained_through'] == 1
    assert second['iterations'][2]['diversity']['previous_elite_n'] == 1


def test_repeated_unchanged_snapshot_does_not_duplicate_log(population):
    read_report(population)
    path = population.output / 'analysis/distribution.log'
    previous = path.read_bytes()
    read_report(population)
    assert path.read_bytes() == previous
    with database.connect(population.db_path) as conn:
        conn.execute('UPDATE compound SET primg_gbsa_score=-20 WHERE id=3')
    result = read_report(population)
    assert result['iterations'][1]['cohorts']['primg_gbsa_scored']['n'] == 2
    assert path.read_bytes().startswith(previous) and path.read_bytes() != previous


def test_nonfinite_scores_are_missing_not_serialized_as_nan(population):
    with database.connect(population.db_path) as conn:
        conn.execute('UPDATE compound SET primg_gbsa_score=?,docking_score=? WHERE id=3', (float('inf'), float('-inf')))
    result = read_report(population)
    stats = result['iterations'][1]['cohorts']['all']['metrics']['docking_score']
    assert stats['n'] == 1 and stats['missing'] == 3
    assert result['iterations'][1]['cohorts']['primg_gbsa_scored']['n'] == 1


def test_invalid_smiles_and_empty_or_reference_only_populations(population):
    with database.connect(population.db_path) as conn:
        conn.execute("UPDATE compound SET smiles='invalid' WHERE id=3")
    result = read_report(population)
    assert 3 in result['invalid_sample_ids']
    assert result['population_n'] == 5
    (population.output / 'analysis/chemical_space_pca.json').unlink()
    result = read_report(population, 0)
    assert result['map_n'] == 1
    assert not (population.output / 'analysis/chemical_space_pca.json').exists()
    with database.connect(population.db_path) as conn:
        conn.execute('DELETE FROM compound')
    result = read_report(population)
    assert result['population_n'] == 0 and result['points'] == []


def test_report_failure_retains_published_html_and_log(population, monkeypatch):
    read_report(population)
    root = population.output / 'analysis'
    before = {name: (root / name).read_bytes() for name in ('index.html', 'summary.json', 'distribution.log')}
    with database.connect(population.db_path) as conn:
        conn.execute('UPDATE compound SET primg_gbsa_score=-26 WHERE id=2')
    def fail(*args):
        raise OSError('cannot render')
    monkeypatch.setattr(reports, 'build_report', fail)
    with pytest.raises(OSError, match='cannot render'):
        read_report(population)
    assert {name: (root / name).read_bytes() for name in before} == before


def test_missing_published_report_is_rebuilt_on_same_snapshot(population):
    read_report(population)
    path = population.output / 'analysis/index.html'
    path.unlink()
    read_report(population)
    assert path.is_file()


def test_sampling_is_balanced_bounded_and_independent_of_scores():
    groups = {1: [{'id': i, 'smiles': 'C'} for i in range(100)],
              2: [{'id': 100 + i, 'smiles': 'N'} for i in range(100)]}
    chosen = reports.balanced_sample(groups, 20)
    assert sum(row['id'] < 100 for row in chosen) == 10
    for group in groups.values():
        for row in group:
            row['primg_gbsa_score'] = -row['id']
    assert reports.balanced_sample(groups, 20) == chosen
    assert reports.diversity([{'id': 1, 'smiles': 'CC'}], [])['internal_diversity'] is None


def test_html_embedded_data_cannot_close_script_tag():
    document = render({'name': '</script><script>alert(1)</script>'})
    assert '</script><script>alert(1)' not in document
    assert r'\u003c/script>' in document
    assert '<script src=' not in document  # Offline operation; no CDN.


def test_manual_cli_needs_no_calculator_inputs_or_project_initialization(population, monkeypatch):
    toml = population.output / 'project.toml'
    toml.write_text('out-dir = "."\n')
    monkeypatch.setattr(_core, 'configure_project', lambda *args: pytest.fail('initialized report source'))
    cli.main(['report', str(toml), '--iteration', '1'])
    assert (population.output / 'analysis/index.html').exists()
    with pytest.raises(FileNotFoundError):
        config.distribution_project(population.output / 'missing.toml')


@pytest.mark.parametrize('settings', [{'distribution-report': 'yes'}, {'distribution-report': 1}])
def test_report_configuration_rejects_invalid_values(settings):
    with pytest.raises(ValueError):
        config.distribution_settings(settings)


def test_old_read_only_schema_has_nullable_diagnostic_columns(tmp_path):
    path = tmp_path / 'old.sqlite'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE compound(id INTEGER,name TEXT,smiles TEXT,iteration INTEGER,state TEXT)')
        conn.execute("INSERT INTO compound VALUES (1,'old','C',1,'generated')")
    rows, scaffolds = database.distribution_snapshot(path)
    assert rows[0]['primg_gbsa_score'] is None and rows[0]['rdk_mw'] is None
    assert scaffolds == {}
    with sqlite3.connect(path) as conn:
        assert len(conn.execute('PRAGMA table_info(compound)').fetchall()) == 5


def test_driver_report_failure_does_not_gate_scientific_work(tmp_path, monkeypatch, capsys):
    args = SimpleNamespace(distribution_report_enabled=True)
    calls = []
    monkeypatch.setattr(driver, 'stream_process', lambda *a: calls.append(a) or 1)
    driver.refresh_distribution_report(args, tmp_path / 'project.toml', 2, 'glide-complete', driver.StopFlag())
    assert calls[0][0] == 'report'
    assert calls[0][1][-4:] == ['--iteration', '2', '--event', 'glide-complete']
    assert 'supervision continues' in capsys.readouterr().out
    args.distribution_report_enabled = False
    driver.refresh_distribution_report(args, tmp_path / 'project.toml', 2, 'glide-complete', driver.StopFlag())
    assert len(calls) == 1


@pytest.mark.parametrize('stage', ['generate', 'druglike', 'synthetic', 'ligprep', 'glide',
                                 'amber-gbsa', 'primg_gbsa', 'primg_gbsa-recovery', 'fep'])
def test_supervisor_reports_only_verified_generate_completion(population, monkeypatch, stage):
    args = population
    args.distribution_report_enabled = True
    args.target_count = 4
    args.primg_gbsa_input_count = 1
    args.synthetic_feasibility_enabled = False
    events = []
    calls = []
    monkeypatch.setattr(driver, 'refresh_distribution_report', lambda *a: events.append(a[3]))
    monkeypatch.setattr(driver, 'stream_process', lambda *a: calls.append(a[0]) or 0)
    pending = iter(['processing', 'processing', None])
    monkeypatch.setattr(database, 'stage_completion_reason', lambda *a: next(pending))
    monkeypatch.setattr(driver, 'external_completion_reason', lambda *a: None)
    class Stop(driver.StopFlag):
        def wait(self, seconds):
            pass
    assert driver.supervise_iteration(args, args.output / 'project.toml', 1,
                                      [(stage, 'module')], 1, Stop())
    assert calls == [stage, stage, stage]
    assert events == (['generate-complete'] if stage == 'generate' else [])
