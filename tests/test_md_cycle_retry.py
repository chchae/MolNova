"""One MD retry per cycle; mocks validate ownership/recovery, not MD energies."""
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database, driver, md_gbsa, schrodinger_guard as guard
from molnova.paths import stage_directory
from molnova.states import CompoundState as State


@pytest.fixture
def project(tmp_path, monkeypatch):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany("INSERT INTO compound(name,smiles,iteration,state,docking_score,prime_gbsa_score) "
                         "VALUES (?,'C',1,'prime_gbsa_done',-10,?)", [('one', -60), ('two', -50)])
    settings = dict(protocol=dict(seed=1, snapshots=1), replicates=1, root='/frozen',
                    ssh_host='slurm', suite='/suite', helper='/helper', gpus=2)
    database.freeze_md_gbsa_selection(db, 1, settings, count=2, target_count=2, prime_count=2)
    args = SimpleNamespace(db_path=db, output=tmp_path, schrodinger=Path('/suite'),
                           md_gbsa_enabled=True, prime_gbsa_enabled=True,
                           target_count=2, prime_gbsa_input_count=2, max_iteration=3,
                           synthetic_feasibility_enabled=False, fep_enabled=True, fep_input_count=2)
    monkeypatch.setattr(md_gbsa, 'prepare', lambda *a: None)
    monkeypatch.setattr(guard, 'register_md_remote', lambda *a: None)
    monkeypatch.setattr(guard, 'unregister_md_remote', lambda *a: None)
    monkeypatch.setattr(guard, 'submission_guard', lambda *a, **kw: nullcontext())
    monkeypatch.setattr(guard, 'active_jobs', lambda *a: [])
    return args, settings


def ensemble(cid, score=-70):
    return dict(id=cid, mean=score, std=0, snapshots=[dict(frame=0, score=score, ligand_rmsd=.2)])


def test_explicit_retry_archives_consumed_token_and_preserves_success(project):
    args,_=project
    database.enable_md_cycle_retries(args.db_path,1)
    database.store_md_gbsa_result(args.db_path,1,2,ensemble(2))
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='failed',failed_stage='md_gbsa' WHERE id=1")
    database.request_md_cycle_retries(args.db_path,1)
    old=database.score_retry_requests(args.db_path,1,'md_gbsa')[0][1]
    database.finish_score_retry(args.db_path,1,'md_gbsa',1,old)
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='failed',failed_stage='md_gbsa' WHERE id=1")
    tokens=database.request_authorized_md_retries(args.db_path,1,[1])
    assert tokens[1]!=old
    assert database.request_authorized_md_retries(args.db_path,1,[1])==tokens
    assert database.md_gbsa_rows(args.db_path,1)[2][1]==-70
    with pytest.raises(ValueError,match='only unscored failed'):
        database.request_authorized_md_retries(args.db_path,1,[1,2])
    with database.connect(args.db_path) as conn:
        saved=conn.execute('SELECT previous_record FROM md_manual_retry_history').fetchone()[0]
        assert json.loads(saved)[3]==old


@pytest.mark.parametrize('retry_success', [False, True])
def test_retry_resolves_inside_cycle_and_never_reopens_on_restart(project, monkeypatch, retry_success):
    args, settings = project
    polls = {1: 0, 2: 0}
    resets = []
    monkeypatch.setattr(md_gbsa, 'ssh', lambda *a, **kw: resets.append(a) or '{"status":"ready"}')
    def poll(config, action, root):
        cid = int(root.split('/CMPID_')[1].split('/')[0])
        polls[cid] += 1
        if cid == 1 and (polls[cid] == 1 or not retry_success):
            answer = dict(status='failed', phase='md', error='verified terminal native failure')
        else:
            answer = dict(ensemble(cid), status='done', protocol=settings['protocol'])
        return 'MOLNOVA_MD=' + json.dumps(answer)
    monkeypatch.setattr(md_gbsa, 'remote_call', poll)
    md_gbsa.run(args, 1)
    assert database.score_retry_requests(args.db_path, 1, 'md_gbsa')
    assert database.md_gbsa_completion_reason(args.db_path, 1)
    with pytest.raises(RuntimeError, match='waiting'):
        database.fep_candidates(args.db_path, 1, 2, target_count=2, prime_count=2)
    token = database.score_retry_requests(args.db_path, 1, 'md_gbsa')[0][1]
    # Restart while recovery is pending preserves the same intent/token.
    database.request_latest_score_retries(args)
    assert database.score_retry_requests(args.db_path, 1, 'md_gbsa') == [(1, token)]
    md_gbsa.run(args, 1)
    assert len(resets) == 1
    assert database.md_gbsa_completion_reason(args.db_path, 1) is None
    rows = database.md_gbsa_rows(args.db_path, 1)
    assert rows[2][1] == -70
    assert rows[1][2] == (State.MD_GBSA_DONE if retry_success else State.FAILED)
    fep = database.fep_candidates(args.db_path, 1, 2, target_count=2, prime_count=2)
    assert len(fep) == (2 if retry_success else 1)
    for _ in range(3):
        assert database.request_latest_score_retries(args) is None
        md_gbsa.run(args, 1)
    assert polls == {1: 2, 2: 1}
    assert len(resets) == 1
    assert driver.choose_iteration(args) == 2
    audit = stage_directory(args.output, 1, 'md_gbsa') / 'retry_history' / token / 'progress.json'
    assert json.loads(audit.read_text())['(1, 0)']['status'] == 'failed'


def test_failed_state_before_retry_request_cannot_finish_or_export(project):
    args, _ = project
    database.enable_md_cycle_retries(args.db_path, 1)
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='failed',failed_stage='md_gbsa' WHERE id=1")
    database.store_md_gbsa_result(args.db_path, 1, 2, ensemble(2))
    assert 'in-cycle retry' in database.md_gbsa_completion_reason(args.db_path, 1)
    with pytest.raises(RuntimeError, match='in-cycle retry'):
        database.fep_candidates(args.db_path, 1, 2)
    database.request_md_cycle_retries(args.db_path, 1)
    assert len(database.score_retry_requests(args.db_path, 1, 'md_gbsa')) == 1


def test_one_driver_invocation_waits_for_retry_before_finishing_and_exporting(project, monkeypatch):
    args, settings = project
    from molnova.stages import fep
    calls = []
    md_polls = []
    monkeypatch.setattr(driver, 'external_completion_reason', lambda *a: None)
    monkeypatch.setattr(md_gbsa, 'ssh', lambda *a, **kw: '{"status":"ready"}')
    def poll(config, action, root):
        cid = int(root.split('/CMPID_')[1].split('/')[0])
        md_polls.append(cid)
        if cid == 1:
            answer = dict(status='failed', phase='md', error='verified terminal failure')
        else:
            answer = dict(ensemble(cid), status='done', protocol=settings['protocol'])
        return 'MOLNOVA_MD=' + json.dumps(answer)
    monkeypatch.setattr(md_gbsa, 'remote_call', poll)
    def process(stage, *unused):
        calls.append(stage)
        if stage == 'md-gbsa':
            with database.connect(args.db_path) as conn:
                assert conn.execute('SELECT finished_at FROM stage_timing').fetchone() == (None,)
            md_gbsa.run(args, 1)
        else:
            assert calls == ['md-gbsa', 'md-gbsa', 'fep']
            assert database.md_gbsa_completion_reason(args.db_path, 1) is None
            fep.export_candidates(args, 1)
        return 0
    monkeypatch.setattr(driver, 'stream_process', process)
    class Stop(driver.StopFlag):
        def wait(self, seconds):
            pass
    assert driver.supervise_iteration(args, args.output/'project.toml', 1,
        [('md-gbsa', 'molnova.stages.md_gbsa'), ('fep', 'molnova.stages.fep')], 1, Stop())
    assert calls == ['md-gbsa', 'md-gbsa', 'fep']
    assert md_polls == [1, 2, 1]
    assert (stage_directory(args.output, 1, 'fep')/'fep_candidates.tsv').is_file()


def test_legacy_live_retry_is_adopted_without_third_attempt_and_early_finish_is_audited(project):
    args, _ = project
    database.store_md_gbsa_result(args.db_path, 1, 2, ensemble(2))
    with database.connect(args.db_path) as conn:
        database._ensure_score_retry_schema(conn)
        conn.execute("INSERT INTO score_restart_retry VALUES (1,'md_gbsa',1,'existing-token',0,'failed','old failure')")
    database.start_iteration_timer(args.db_path, 1)
    database.start_stage_timer(args.db_path, 1, 'md-gbsa')
    database.finish_stage_timer(args.db_path, 1, 'md-gbsa')
    database.finish_iteration_timer(args.db_path, 1)
    database.enable_md_cycle_retries(args.db_path, 1)
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT finished_at FROM iteration_timing').fetchone() == (None,)
        assert conn.execute("SELECT finished_at FROM stage_timing WHERE stage='md-gbsa'").fetchone() == (None,)
        assert conn.execute('SELECT COUNT(*) FROM score_completion_history').fetchone() == (2,)
        assert conn.execute('SELECT token,pending FROM score_restart_retry').fetchone() == ('existing-token', 0)
        conn.execute("UPDATE compound SET state='failed',failed_stage='md_gbsa' WHERE id=1")
    assert database.request_latest_score_retries(args) is None
    assert database.md_gbsa_completion_reason(args.db_path, 1) is None


@pytest.mark.parametrize('imported', [False, True])
def test_early_fep_selection_refreshed_after_resolution_or_protected_if_imported(project, imported):
    args, _ = project
    from molnova.stages import fep
    database.store_md_gbsa_result(args.db_path, 1, 2, ensemble(2, -70))
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='failed',failed_stage='md_gbsa' WHERE id=1")
    assert [r['id'] for r in database.fep_candidates(args.db_path, 1, 2)] == [2]
    outfile = fep.export_candidates(args, 1)
    original_tsv = outfile.read_bytes()
    if imported:
        with database.connect(args.db_path) as conn:
            conn.execute('UPDATE compound SET fep_score=-5 WHERE id=2')
    database.enable_md_cycle_retries(args.db_path, 1)
    # Imports still refer to the exact historical selection during recovery.
    assert [r['id'] for r in database.fep_candidates(args.db_path, 1, 2, for_import=True)] == [2]
    cid, token = database.score_retry_requests(args.db_path, 1, 'md_gbsa')[0]
    database.finish_score_retry(args.db_path, 1, 'md_gbsa', cid, token)
    database.store_md_gbsa_result(args.db_path, 1, 1, ensemble(1, -90))
    if imported:
        with pytest.raises(RuntimeError, match='imported scores'):
            database.fep_candidates(args.db_path, 1, 2)
        return
    assert [r['id'] for r in database.fep_candidates(args.db_path, 1, 2)] == [1, 2]
    fep.export_candidates(args, 1)
    assert outfile.read_bytes() != original_tsv
    assert next(outfile.parent.glob('*.previous.tsv')).read_bytes() == original_tsv
    with database.connect(args.db_path) as conn:
        history = conn.execute('SELECT candidates FROM iteration_fep_selection_history').fetchone()[0]
        assert [r['id'] for r in json.loads(history)] == [2]
    database.fep_candidates(args.db_path, 1, 1)  # A later count change cannot rewrite the new frozen selection.
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM iteration_fep_selection_history').fetchone() == (1,)
