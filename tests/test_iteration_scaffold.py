import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from rdkit import Chem

from molnova import _core, database
from molnova.chemistry.scaffold import prepare_iteration_scaffold, log_iteration_mcs
from molnova.stages import generate, glide


@pytest.fixture
def project(tmp_path):
    path = tmp_path / 'project.sqlite'
    with database.connect(path) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany(
            'INSERT INTO compound(name,smiles,iteration,state) VALUES (?,?,0,\'reference\')',
            [('R1', 'COc1ccc(CC)cc1'), ('R2', 'NCc1ccc(Cl)cc1'),
             ('R3', 'CCOc1ccc(CN)cc1')],
        )
        conn.executemany(
            'INSERT INTO compound(name,smiles,iteration,state,docking_score,gbsa_score) '
            'VALUES (?,?,1,\'gbsa_done\',-10,?)',
            [('E1', 'COc1ccc(CC)cc1', -40), ('E2', 'CCOc1ccc(CN)cc1', -30),
             ('OUTSIDE', 'NCc1ccc(Cl)cc1', -20)],
        )
    return SimpleNamespace(db_path=path, output=tmp_path, automatic_mcs=True,
                           mcs_timeout=60, gbsa_elite_count=2, target_count=3,
                           gbsa_input_count=3, max_iteration=3, sample_size=10,
                           elite_tl_epochs=3, libinvent_prior=tmp_path / 'prior')


def test_reference_first_then_previous_elites_and_restart(project, monkeypatch):
    # Cycle 1 is new in this isolated DB after removing the later test population.
    with database.connect(project.db_path) as conn:
        conn.execute('UPDATE compound SET iteration=2 WHERE iteration=1')
    prepare_iteration_scaffold(project, 1, create=True)
    first_core = project.generation_mcs_smarts
    first = database.iteration_scaffold(project.db_path, 1)
    assert first['source'] == 'iteration0_references'
    assert first['source_ids'] == [1, 2, 3]
    assert '[*:0]' in project.libinvent_scaffold_smiles
    assert '[*:1]' in project.libinvent_scaffold_smiles

    elites = database.iteration_gbsa_elites(project.db_path, 2, 2)
    prepare_iteration_scaffold(project, 3, elites, create=True)
    later = database.iteration_scaffold(project.db_path, 3)
    assert later['mcs_smarts'] != first_core
    assert later['source'] == 'previous_iteration_elites'
    assert later['source_ids'] == [row[0] for row in elites]
    assert all(Chem.MolFromSmiles(row[2]).HasSubstructMatch(
        Chem.MolFromSmarts(later['mcs_smarts'])) for row in elites)

    # Neither changed scores nor missing derivative files change a saved core.
    with database.connect(project.db_path) as conn:
        conn.execute('UPDATE compound SET gbsa_score=-100 WHERE name=\'OUTSIDE\'')
    project.libinvent_scaffold_file.unlink()
    monkeypatch.setattr(_core, 'prepare_libinvent_scaffold',
                        lambda *a, **k: pytest.fail('MCS recomputed during recovery'))
    restarted = SimpleNamespace(**vars(project))
    prepare_iteration_scaffold(restarted, 3)
    assert restarted.generation_mcs_smarts == later['mcs_smarts']
    assert restarted.mcs_smarts == first_core
    assert restarted.libinvent_scaffold_file.read_text().strip() == later['scaffold_smiles']
    assert json.loads((project.output / 'iter3/scaffold/scaffold_info.json').read_text()) == later


@pytest.mark.parametrize('supplied', [None, '[*:0]c1ccc([*:1])cc1'])
def test_generation_passes_ranked_elites_and_new_scaffold_to_tl(project, monkeypatch, supplied):
    project.iteration_mcs = True
    project.initial_mcs_smarts = supplied
    project.automatic_mcs = supplied is None
    seen = []
    def training(elites, scaffold, directory, **kwargs):
        snapshot = database.iteration_scaffold(project.db_path, 2)
        assert snapshot['source_ids'] == [row[0] for row in elites]
        assert scaffold == snapshot['scaffold_smiles']
        if supplied:
            assert snapshot['docking_mcs_smarts'] == supplied
            assert scaffold != supplied
        seen.append('training')
        return None, 'train', 'valid', 2
    monkeypatch.setattr(_core, 'write_elite_libinvent_training_files', training)
    monkeypatch.setattr(_core, 'run_libinvent_elite_transfer_learning', lambda **k: 'model')
    def sample(**kwargs):
        assert kwargs['scaffold_file'] == project.output / 'iter2/scaffold/libinvent_scaffold.smi'
        assert kwargs['libinvent_prior'] == 'model'
        seen.append('sampling')
        return 'generated.csv'
    monkeypatch.setattr(_core, 'run_libinvent_sampling', sample)
    monkeypatch.setattr(_core, 'insert_generated', lambda **k: 3)
    generate._run_stage(project, 2)
    assert seen == ['training', 'sampling']


def test_first_generation_uses_reference_core_and_original_prior(project, monkeypatch):
    with database.connect(project.db_path) as conn:
        conn.execute('UPDATE compound SET iteration=2 WHERE iteration=1')
    def sample(**kwargs):
        assert kwargs['libinvent_prior'] == project.libinvent_prior
        assert kwargs['scaffold_file'] == project.output / 'iter1/scaffold/libinvent_scaffold.smi'
        assert database.iteration_scaffold(project.db_path, 1)['source'] == 'iteration0_references'
        return 'generated.csv'
    monkeypatch.setattr(_core, 'run_libinvent_sampling', sample)
    monkeypatch.setattr(_core, 'run_libinvent_elite_transfer_learning',
                        lambda **k: pytest.fail('TL in first cycle'))
    monkeypatch.setattr(_core, 'insert_generated', lambda **k: 3)
    generate._run_stage(project, 1)


def test_glide_uses_initial_constraint_instead_of_elite_core(project, monkeypatch):
    prepare_iteration_scaffold(project, 2, database.iteration_gbsa_elites(project.db_path, 1, 2),
                               create=True)
    core = project.mcs_smarts
    assert project.generation_mcs_smarts != core
    project.mcs_smarts = 'invalid old core'
    monkeypatch.setattr(database, 'upstream_completion_reason', lambda *a, **k: None)
    def assign(iteration, args):
        assert iteration == 2
        assert args.mcs_smarts == core
    monkeypatch.setattr(_core, 'assign_reference_compounds', assign)
    monkeypatch.setattr(glide, 'query_groups', lambda *a: {})
    glide._run_stage(project, SimpleNamespace(iteration=2, poll_interval=1,
                                             completion_fraction=0.95, tail_timeout=None))


def test_explicit_initial_mcs_still_adapts_generation_to_elites(project):
    project.iteration_mcs = True
    project.automatic_mcs = False
    project.initial_mcs_smarts = '[*:0]c1ccc([*:1])cc1'
    with database.connect(project.db_path) as conn:
        conn.execute('UPDATE compound SET iteration=2 WHERE iteration=1')
    prepare_iteration_scaffold(project, 1, create=True)
    assert project.libinvent_scaffold_smiles == project.initial_mcs_smarts
    assert database.iteration_scaffold(project.db_path, 1)['source'] == 'user_initial_mcs'
    prepare_iteration_scaffold(project, 3, database.iteration_gbsa_elites(project.db_path, 2, 2),
                               create=True)
    assert project.mcs_smarts == project.initial_mcs_smarts
    assert project.generation_mcs_smarts != project.initial_mcs_smarts
    assert database.iteration_scaffold(project.db_path, 3)['source'] == 'previous_iteration_elites'


def test_incomplete_iteration_cannot_compute_new_core(project, monkeypatch):
    with database.connect(project.db_path) as conn:
        conn.execute('UPDATE compound SET state=\'glide_running\' WHERE name=\'OUTSIDE\'')
    monkeypatch.setattr(_core, 'prepare_libinvent_scaffold', lambda *a, **k: pytest.fail('gate bypass'))
    generate._run_stage(project, 2)
    assert database.iteration_scaffold(project.db_path, 2) is None


def test_elite_scaffold_missing_initial_constraint_stops_before_generation(project):
    elites = [(100, 'A', 'CCCn1cc(CC)cc1', 1, -10, -40),
              (101, 'B', 'NCCn1cc(CN)cc1', 1, -9, -30)]
    with pytest.raises(RuntimeError, match='does not contain the fixed docking core'):
        prepare_iteration_scaffold(project, 2, elites, create=True)
    assert database.iteration_scaffold(project.db_path, 2) is None


def test_elite_core_can_expand_beyond_original_pose_core(project):
    elites = [(100, 'A', 'CCCOc1ccc(CNCC)cc1', 1, -10, -40),
              (101, 'B', 'NCCOc1ccc(CNCO)cc1', 1, -9, -30)]
    prepare_iteration_scaffold(project, 2, elites, create=True)
    generation_query = Chem.MolFromSmarts(project.generation_mcs_smarts)
    references = _core.get_reference_compounds(project.db_path)
    assert not any(ref['mol'].HasSubstructMatch(generation_query) for ref in references)
    assert any(ref['mol'].HasSubstructMatch(Chem.MolFromSmarts(project.mcs_smarts))
               for ref in references)
    assert all(_core.decompose_elite_for_libinvent(row[2], project.libinvent_scaffold_smiles)
               is not None for row in elites)


def test_old_iteration_keeps_its_already_used_docking_core(project, monkeypatch):
    elites = database.iteration_gbsa_elites(project.db_path, 1, 2)
    inputs = [dict(id=r[0], name=r[1], mol=Chem.MolFromSmiles(r[2])) for r in elites]
    info = _core.prepare_libinvent_scaffold(inputs, None, project.output / 'old')
    old = dict(mcs_smarts=info['mcs_smarts'], scaffold_smiles=info['scaffold_smiles'],
               source='previous_iteration_elites', source_ids=[r[0] for r in elites])
    database.iteration_scaffold(project.db_path, 2, old)
    monkeypatch.setattr(_core, 'prepare_libinvent_scaffold',
                        lambda *a, **k: pytest.fail('existing calculation changed'))
    prepare_iteration_scaffold(project, 2)
    assert project.mcs_smarts == old['mcs_smarts']
    assert database.iteration_scaffold(project.db_path, 2) == old


def test_existing_iteration_keeps_legacy_scaffold(project, monkeypatch):
    (project.output / 'mcs.smarts').write_text('c1ccccc1\n')
    (project.output / 'libinvent_scaffold.smi').write_text('[*:0]c1ccc([*:1])cc1\n')
    monkeypatch.setattr(_core, 'prepare_libinvent_scaffold', lambda *a, **k: pytest.fail('live core changed'))
    prepare_iteration_scaffold(project, 1, create=True)
    assert project.mcs_smarts == 'c1ccccc1'
    assert database.iteration_scaffold(project.db_path, 1)['source'] == 'legacy_project_scaffold'
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT gbsa_score FROM compound WHERE name=\'E1\'').fetchone()[0] == -40


def test_missing_legacy_core_does_not_recompute(project):
    with pytest.raises(RuntimeError, match='restore the original'):
        prepare_iteration_scaffold(project, 1, create=True)
    assert database.iteration_scaffold(project.db_path, 1) is None


@pytest.mark.parametrize('failure', ['timeout', 'single_input', 'no_attachments'])
def test_mcs_failure_does_not_save_core_or_change_results(project, monkeypatch, failure):
    elites = database.iteration_gbsa_elites(project.db_path, 1, 2)
    if failure == 'timeout':
        monkeypatch.setattr(_core.rdFMCS, 'FindMCS',
                            lambda *a, **k: SimpleNamespace(canceled=True))
        message = 'timed out'
    elif failure == 'single_input':
        project.gbsa_elite_count = 1
        elites = elites[:1]
        message = 'At least two input ligands'
    else:
        elites = [(100, 'A', 'c1ccccc1', 1, -10, -40),
                  (101, 'B', 'c1ccccc1', 1, -9, -30)]
        message = 'Fewer than two variable positions'
    with pytest.raises(RuntimeError, match=message):
        prepare_iteration_scaffold(project, 2, elites, create=True)
    assert database.iteration_scaffold(project.db_path, 2) is None
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT gbsa_score,state FROM compound WHERE name=\'E1\'').fetchone() == (-40, 'gbsa_done')


def test_scaffold_snapshot_is_atomic_and_immutable(project):
    def save(index):
        return database.iteration_scaffold(project.db_path, 2, {'source': index})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, [1, 2]))
    assert results[0] == results[1] == database.iteration_scaffold(project.db_path, 2)


@pytest.mark.parametrize('supplied', [None, '[*:0]c1ccc([*:1])cc1'])
def test_project_initialization_defers_core_calculations(project, monkeypatch, supplied):
    cfg = dict(project='project', toml_file=project.output / 'project.toml',
               db_path=project.db_path, output=project.output, log_level=1,
               grid=project.output / 'grid.zip', reference_poses=project.output / 'refs.maegz',
               libinvent_prior=project.libinvent_prior, mcs_smarts=supplied)
    monkeypatch.setattr(_core, 'load_project_toml', lambda path: cfg)
    monkeypatch.setattr(_core, 'initialize_project_database', lambda **kwargs: None)
    monkeypatch.setattr(_core, 'get_reference_compounds', lambda *a: pytest.fail('core outside worker'))
    args = _core.configure_project(cfg['toml_file'])
    assert args.iteration_mcs
    assert args.initial_mcs_smarts == supplied
    assert args.mcs_smarts == supplied


@pytest.mark.parametrize('supplied', [None, '[*:0]c1ccc([*:1])cc1'])
def test_default_log_shows_initial_and_elite_cores_on_create_and_resume(
        project, monkeypatch, capsys, supplied):
    from molnova.stages import _logging
    token = _logging.set_log_level(1)
    try:
        project.iteration_mcs = True
        project.initial_mcs_smarts = supplied
        prepare_iteration_scaffold(project, 2,
                                   database.iteration_gbsa_elites(project.db_path, 1, 2),
                                   create=True)
        initial = database.iteration_scaffold(project.db_path, 0)['mcs_smarts']
        generation = project.generation_mcs_smarts
        assert initial != generation
        output = capsys.readouterr().out
        assert f'[MCS] Initial docking constraint: {initial}' in output
        assert f'[MCS] Iteration 2 generation core (iteration 1 elites): {generation}' in output
        monkeypatch.setattr(_core, 'prepare_libinvent_scaffold',
                            lambda *a, **k: pytest.fail('logging recalculated MCS'))
        prepare_iteration_scaffold(project, 2)
        resumed_output = capsys.readouterr().out
        assert f'[MCS] Initial docking constraint: {initial}' in resumed_output
        assert f'[MCS] Iteration 2 generation core (iteration 1 elites): {generation}' in resumed_output
    finally:
        _logging._log_level.reset(token)


def test_finished_generation_still_logs_saved_cores_without_recalculation(project, monkeypatch, capsys):
    (project.output / 'mcs.smarts').write_text('c1ccccc1\n')
    (project.output / 'libinvent_scaffold.smi').write_text('[*:0]c1ccc([*:1])cc1\n')
    prepare_iteration_scaffold(project, 1, create=True)
    capsys.readouterr()
    monkeypatch.setattr(_core, 'prepare_libinvent_scaffold',
                        lambda *a, **k: pytest.fail('completed generation recalculated core'))
    generate._run_stage(project, 1)
    output = capsys.readouterr().out
    assert '[MCS] Initial docking constraint: c1ccccc1' in output
    assert '[MCS] Iteration 1 generation core (saved legacy scaffold): c1ccccc1' in output
    assert 'generation skipped' in output


def test_legacy_log_distinguishes_initial_core_from_preserved_glide_core(project, capsys):
    database.iteration_scaffold(project.db_path, 0, dict(mcs_smarts='c1ccccc1'))
    legacy = dict(mcs_smarts='Cc1ccccc1', source='previous_iteration_elites')
    log_iteration_mcs(project, 2, legacy)
    output = capsys.readouterr().out
    assert '[MCS] Initial docking constraint: c1ccccc1' in output
    assert '[MCS] Iteration 2 preserved Glide constraint: Cc1ccccc1' in output
