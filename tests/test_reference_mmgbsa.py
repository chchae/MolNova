import json
import runpy
import sqlite3
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from molnova import _core as c, database, driver, gbsa_queue, prime_async
from molnova.stages import mmgbsa


def project(tmp_path):
    db = tmp_path / 'ref.sqlite'
    with sqlite3.connect(db) as conn:
        c.create_sqlite_schema(conn)
        conn.executemany(
            "INSERT INTO compound(name,smiles,iteration,state,gbsa_score) VALUES (?,'C',0,'reference',?)",
            [('existing', -40), ('missing', None), ('missing2', None)],
        )
    return SimpleNamespace(db_path=db, output=tmp_path, gbsa_input_count=1,
                           target_count=1000, max_iteration=20)


def test_references_ignore_docking_count_and_target_and_preserve_existing_scores(tmp_path):
    args = project(tmp_path)
    assert database.mmgbsa_candidates(args.db_path, 0, 1, 1000) == [2, 3]
    assert mmgbsa.eligible_iteration(args) == 0
    assert driver.choose_iteration(args) == 0
    assert driver.choose_iteration(args, 0) == 0
    assert database.claim_mmgbsa_candidates(args.db_path, 0, 1, 1000) == [2, 3]
    assert database.claim_mmgbsa_candidates(args.db_path, 0, 1, 1000) == []
    assert 'running for 2' in database.stage_completion_reason(args.db_path, 0, 'mmgbsa', 1000, 1)
    scores = tmp_path / 'scores.tsv'
    scores.write_text('id\tgbsa_score\n1\t-99\n2\t-35\n3\t-30\n')
    assert c.update_gbsa_scores(scores, args.db_path, allowed_ids=[1, 2, 3]) == 2
    assert database.reference_mmgbsa_completion_reason(args.db_path) is None
    assert len(c.get_reference_compounds(args.db_path)) == 3
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute('SELECT gbsa_score FROM compound ORDER BY id').fetchall() == [(-40,), (-35,), (-30,)]


@pytest.mark.parametrize('requested', [None, 0])
def test_worker_handles_zero_explicitly_and_only_calculates_missing_references(tmp_path, monkeypatch, requested):
    args = project(tmp_path)
    calls = []
    def calculate(**kw):
        calls.append(kw)
        assert kw['iteration'] == 0 and kw['compound_ids'] == [2, 3]
        assert kw['iteration_dir'] == tmp_path / 'iter0'
        scores = tmp_path / 'scores.tsv'
        scores.write_text('id\tgbsa_score\n2\t-35\n3\t-30\n')
        return c.update_gbsa_scores(scores, args.db_path, allowed_ids=kw['compound_ids'])
    monkeypatch.setattr(c, 'run_iteration_mmgbsa', calculate)
    mmgbsa._run_stage(args, requested)
    assert len(calls) == 1
    mmgbsa._run_stage(args, requested)
    assert len(calls) == 1


def test_orphan_reference_claim_restores_reference_and_protects_saved_ids(tmp_path):
    args = project(tmp_path)
    database.claim_mmgbsa_candidates(args.db_path, 0, 1, 1000)
    assert database.reset_unsubmitted_mmgbsa(args.db_path, 0, [2]) == [3]
    rows = database.mmgbsa_submission_rows(args.db_path, [2, 3])
    assert rows[2][1] == 'gbsa_running'
    assert rows[3][1] == 'reference'


def test_reference_terminal_failure_does_not_block_forever(tmp_path):
    args = project(tmp_path)
    database.claim_mmgbsa_candidates(args.db_path, 0, 1, 1000)
    database.mark_gbsa_atomtyping_failures(args.db_path, [2, 3], 'Cannot atom type structure')
    assert database.reference_mmgbsa_completion_reason(args.db_path) is None
    assert database.mmgbsa_candidates(args.db_path, 0, 1, 1000) == []


def test_reference_input_uses_original_pose_and_named_identity_map(tmp_path, monkeypatch):
    args = project(tmp_path)
    args.reference_poses = tmp_path / 'references.maegz'
    args.reference_poses.write_text('poses')
    args.mmgbsa_receptor = tmp_path / 'receptor.maegz'
    args.mmgbsa_receptor.write_text('receptor')
    args.schrodinger = Path('/suite')
    args.grid = tmp_path / 'grid.zip'
    calls = []
    monkeypatch.setattr(c, 'run_command', lambda command, **kw: calls.append(command))
    monkeypatch.setattr(gbsa_queue, 'poll_queue', lambda *a: None)
    assert c.run_iteration_mmgbsa(0, args, tmp_path / 'iter0', tmp_path / 'unused', [2, 3]) is None
    assert calls[0][3] == args.reference_poses
    mapping = json.loads(Path(calls[0][-1]).read_text())
    assert mapping == {'missing': 2, 'missing2': 3}
    saved = prime_async.load_job(tmp_path / 'iter0/mmgbsa')
    assert saved['compound_ids'] == [2, 3]


def test_reference_pv_builder_maps_original_titles_and_does_not_mutate_source(tmp_path, monkeypatch):
    receptor_file, poses_file, ids_file, output_file, mapping = [tmp_path / n for n in ('receptor', 'poses', 'ids', 'output', 'map')]
    ids_file.write_text('2\n3\n')
    mapping.write_text(json.dumps({'missing': 2, 'REF_0003': 3}))
    source = [SimpleNamespace(title='existing', property={}),
              SimpleNamespace(title='missing', property={}), SimpleNamespace(title='', property={})]
    written = []
    class Writer:
        def __init__(self, path): pass
        def append(self, st): written.append(st)
        def close(self): pass
    def reader(path):
        return [SimpleNamespace(title='receptor', property={})] if path == str(receptor_file) else [
            SimpleNamespace(title=s.title, property=dict(s.property)) for s in source]
    module = ModuleType('schrodinger')
    module.structure = SimpleNamespace(StructureReader=reader, StructureWriter=Writer)
    monkeypatch.setitem(sys.modules, 'schrodinger', module)
    script = c.create_mmgbsa_pv_builder(tmp_path)
    monkeypatch.setattr(sys, 'argv', [str(script), *map(str, (receptor_file, poses_file, ids_file, output_file, mapping))])
    runpy.run_path(str(script), run_name='__main__')
    assert [s.title for s in written] == ['receptor', 'CMPID_2', 'CMPID_3']
    assert [s.title for s in source] == ['existing', 'missing', '']
    assert [s.property['i_user_compound_id'] for s in written[1:]] == [2, 3]


def test_new_reference_import_retains_input_gbsa_score(tmp_path, monkeypatch):
    from rdkit import Chem
    suite = tmp_path / 'suite'
    (suite / 'utilities').mkdir(parents=True)
    (suite / 'utilities/structconvert').touch()
    known, missing = Chem.MolFromSmiles('CC'), Chem.MolFromSmiles('CCC')
    known.SetProp('_Name', 'known')
    known.SetDoubleProp('r_psp_MMGBSA_dG_Bind', -42)
    missing.SetProp('_Name', 'missing')
    monkeypatch.setattr(c, 'run_command', lambda *a, **kw: None)
    monkeypatch.setattr(Chem, 'SDMolSupplier', lambda *a, **kw: [known, missing])
    c.import_reference_poses_to_sqlite(tmp_path / 'db.sqlite', tmp_path / 'poses.maegz', suite)
    with sqlite3.connect(tmp_path / 'db.sqlite') as conn:
        assert conn.execute('SELECT name,gbsa_score FROM compound ORDER BY id').fetchall() == [('known', -42), ('missing', None)]


@pytest.mark.parametrize('requested, expected', [(0, [0]), (2, [0, 2])])
def test_driver_reference_stage_runs_without_generation(tmp_path, monkeypatch, requested, expected):
    args = project(tmp_path)
    args.project = 'ref'
    args.synthetic_feasibility_enabled = False
    args.fep_enabled = False
    monkeypatch.setattr(driver.c, 'configure_project', lambda *a: args)
    monkeypatch.setattr(driver.signal, 'signal', lambda *a: None)
    calls = []
    def supervise(args, project, iteration, stages, *other):
        calls.append(iteration)
        if iteration == 0:
            assert stages == [('mmgbsa', 'molnova.stages.mmgbsa')]
            with database.connect(args.db_path) as conn:
                conn.execute("UPDATE compound SET gbsa_score=-30 WHERE iteration=0 AND gbsa_score IS NULL")
        return True
    monkeypatch.setattr(driver, 'supervise_iteration', supervise)
    driver.main([str(tmp_path / 'ref.toml'), '--iteration', str(requested)])
    assert calls == expected


def test_saved_reference_queue_recovers_without_reset_or_resubmission(tmp_path, monkeypatch):
    args = project(tmp_path)
    database.claim_mmgbsa_candidates(args.db_path, 0, 1, 1000)
    root = tmp_path / 'iter0/mmgbsa'
    root.mkdir(parents=True)
    record = gbsa_queue.create_queue(args, root, [2, 3])
    def recover(**kw):
        assert prime_async.load_job(root) == record
        assert kw['iteration'] == 0
        assert all(row[1] == 'gbsa_running' for row in database.mmgbsa_submission_rows(args.db_path, [2, 3]).values())
        return None
    monkeypatch.setattr(c, 'run_iteration_mmgbsa', recover)
    mmgbsa._run_stage(args, 0)
    assert prime_async.load_job(root) == record
