import sqlite3
from types import SimpleNamespace

import pytest

from molnova import _core, database
from molnova.stages import generate
from molnova.states import CompoundState as State


@pytest.fixture
def project(tmp_path):
    db = tmp_path / 'project.sqlite'
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany(
            'INSERT INTO compound(name,smiles,iteration,docking_score,gbsa_score,state) '
            "VALUES (?, 'C', 1, ?, ?, ?)",
            [('BEST', -10, -40, State.GBSA_DONE),
             ('SECOND', -9, -35, State.GBSA_DONE),
             ('OUTSIDE', -8, None, State.DOCKED)],
        )
    return SimpleNamespace(db_path=db, target_count=3, gbsa_input_count=2,
                           gbsa_elite_count=2, max_iteration=4, output=tmp_path)


def reason(args):
    return database.iteration_completion_reason(
        args.db_path, 1, args.target_count, args.gbsa_input_count, args.gbsa_elite_count)


def update(args, name, state, docking_score, gbsa_score):
    with sqlite3.connect(args.db_path) as conn:
        conn.execute('UPDATE compound SET state=?,docking_score=?,gbsa_score=? WHERE name=?',
                     (state, docking_score, gbsa_score, name))


def test_complete_top_n_does_not_require_scores_outside_top_n(project):
    assert reason(project) is None


@pytest.mark.parametrize('state', [State.GENERATED, State.SYNTHETIC_RUNNING,
                                  State.LIGPREP_RUNNING, State.LIGPREPPED, State.GLIDE_RUNNING])
def test_wait_for_late_docking_even_with_enough_elites(project, state):
    update(project, 'OUTSIDE', state, None, None)
    assert 'LigPrep/Glide' in reason(project)
    # A later, better docking result must be scored before generation.
    update(project, 'OUTSIDE', State.DOCKED, -11, None)
    assert 'unfinished MM-GBSA' in reason(project)
    update(project, 'OUTSIDE', State.GBSA_DONE, -11, -50)
    assert reason(project) is None


def test_wait_for_final_top_n_even_with_enough_scores_elsewhere(project):
    update(project, 'OUTSIDE', State.GBSA_DONE, -8, -30)
    update(project, 'SECOND', State.DOCKED, -9, None)
    assert 'unfinished MM-GBSA' in reason(project)


def test_running_batch_outside_final_top_n_blocks_advance(project):
    update(project, 'OUTSIDE', State.GBSA_RUNNING, -8, None)
    assert 'still running' in reason(project)


def test_terminal_failure_processed_but_minimum_elites_required(project):
    update(project, 'SECOND', State.FAILED, -9, None)
    assert 'only 1/2' in reason(project)
    update(project, 'OUTSIDE', State.GBSA_DONE, -8, -30)
    assert reason(project) is None


def test_generation_must_reach_target(project):
    project.target_count = 4
    assert 'generation incomplete' in reason(project)


def test_nonfinite_gbsa_does_not_complete_top_n(project):
    update(project, 'SECOND', State.GBSA_DONE, -9, float('inf'))
    assert 'unfinished MM-GBSA' in reason(project)


@pytest.mark.parametrize('requested', [None, 2])
def test_stage_blocks_tl_and_sampling_until_completed(project, monkeypatch, requested, capsys):
    update(project, 'OUTSIDE', State.GBSA_RUNNING, -8, None)
    def forbidden(*args, **kwargs):
        pytest.fail('TL or sampling started before MM-GBSA completion')
    monkeypatch.setattr(_core, 'run_libinvent_elite_transfer_learning', forbidden)
    monkeypatch.setattr(_core, 'run_libinvent_sampling', forbidden)
    generate._run_stage(project, requested)
    assert 'not ready' in capsys.readouterr().out


def test_elites_are_lowest_scores_from_immediately_previous_iteration(project):
    with sqlite3.connect(project.db_path) as conn:
        conn.executemany(
            'INSERT INTO compound(name,smiles,iteration,docking_score,gbsa_score,state) '
            "VALUES (?, 'C', 2, ?, ?, 'gbsa_done')",
            [('NEW_A', -5, -20), ('NEW_B', -4, -30), ('INVALID', -6, float('-inf'))],
        )
    elites = database.iteration_gbsa_elites(project.db_path, 2, 2)
    assert [(r[1], r[3], r[5]) for r in elites] == [('NEW_B', 2, -30), ('NEW_A', 2, -20)]


def test_completed_stage_uses_previous_elites_before_tl_and_sampling(project, monkeypatch):
    project.libinvent_prior = project.output / 'prior'
    project.libinvent_scaffold_smiles = '[*]C'
    project.libinvent_scaffold_file = project.output / 'scaffold'
    project.elite_tl_epochs = 3
    project.sample_size = 10
    events = []
    def training(elites, scaffold, directory):
        events.append(('elites', [r[1] for r in elites]))
        return None, 'train', 'valid', 2
    def tl(**kwargs):
        events.append(('tl', kwargs['prior_file']))
        return 'model'
    def sampling(**kwargs):
        events.append(('sample', kwargs['libinvent_prior']))
        return 'generated.csv'
    monkeypatch.setattr(_core, 'write_elite_libinvent_training_files', training)
    monkeypatch.setattr(_core, 'run_libinvent_elite_transfer_learning', tl)
    monkeypatch.setattr(_core, 'run_libinvent_sampling', sampling)
    monkeypatch.setattr(_core, 'insert_generated', lambda **kwargs: 3)
    generate._run_stage(project, 2)
    assert events == [('elites', ['BEST', 'SECOND']), ('tl', project.libinvent_prior), ('sample', 'model')]


def test_old_database_cannot_advance_while_earlier_iteration_is_running(project, capsys):
    update(project, 'OUTSIDE', State.GBSA_RUNNING, -8, None)
    with sqlite3.connect(project.db_path) as conn:
        conn.executemany(
            'INSERT INTO compound(name,smiles,iteration,docking_score,gbsa_score,state) '
            "VALUES (?, 'C', 2, -5, -20, 'gbsa_done')", [('A',), ('B',), ('C',)])
    generate._run_stage(project, None)
    assert 'Iteration 3 not ready: iteration 1' in capsys.readouterr().out
