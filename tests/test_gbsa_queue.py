import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from molnova import _core as c, database, prime_async as prime, gbsa_queue as q
from molnova.stages import mmgbsa


def fixture(tmp_path, monkeypatch, count=10):
    from molnova import schrodinger_guard
    monkeypatch.setattr(schrodinger_guard, 'active_jobs', lambda *a: [])
    db = tmp_path / 'results.sqlite'
    with sqlite3.connect(db) as conn:
        c.create_sqlite_schema(conn)
        conn.executemany("INSERT INTO compound(name,smiles,iteration,docking_score,state) "
                         "VALUES (?,'C',1,-9,'gbsa_running')", [(str(i),) for i in range(count)])
    args = SimpleNamespace(db_path=db, output=tmp_path, host='compute:128',
                           schrodinger=Path('/suite'), gbsa_elite_count=1)
    directory = tmp_path / 'iter1/mmgbsa'
    directory.mkdir(parents=True)
    ids = list(range(1, count + 1))
    record = q.create_queue(args, directory, ids)
    record['prepared'] = True
    prime._save(directory, record)
    for cid in ids:
        folder = q._folder(directory, record, cid)
        folder.mkdir(parents=True)
        (folder / 'mmgbsa_input.maegz').write_text('PV')
    launched, completed = [], set()
    def submit(command, folder, compound_ids, **kw):
        cid = compound_ids[0]
        saved = prime.load_job(folder)
        if saved is None:
            launched.append(cid)
            assert command[3:] == ['-HOST', 'compute', '-NJOBS', '1']
            prime._save(folder, {'job_id': f'job-{cid}', 'compound_ids': [cid]})
        return cid in completed
    def finish(iteration, args, folder, compound_ids):
        cid = compound_ids[0]
        score = folder / 'scores.tsv'
        score.write_text(f'id\tgbsa_score\n{cid}\t-35\n')
        updated = c.update_gbsa_scores(score, args.db_path, allowed_ids=compound_ids)
        (folder / 'gbsa_top.tsv').write_text('elites')
        (folder / 'mmgbsa_input-out.maegz').write_text(f'receptor\nCMPID_{cid}\n')
        return updated
    monkeypatch.setattr(prime, 'submit_or_poll', submit)
    monkeypatch.setattr(c, 'finish_iteration_mmgbsa', finish)
    def merge(command, cwd, **kw):
        assert Path(command[1]).name == '_merge_best_poses.py'
        Path(command[2]).write_text(''.join(Path(path).read_text() for path in command[3:]))
    monkeypatch.setattr(c, 'run_command', merge)
    return args, directory, record, launched, completed


def test_refill_one_free_slot_while_six_jobs_still_running(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    assert q.poll_queue(1, args, root, record) is None
    assert launches == list(range(1, 8))
    completed.add(2)
    # Simulate worker restart: recover durable queue and existing JobIds.
    assert q.poll_queue(1, args, root, prime.load_job(root)) is None
    assert launches == list(range(1, 9))
    saved = prime.load_job(root)
    assert saved['entries']['2'] == 'done'
    assert sum(s == 'active' for s in saved['entries'].values()) == 7
    rows = database.mmgbsa_submission_rows(args.db_path, [1, 2, 8])
    assert rows[2][1:] == ('gbsa_done', -35)
    assert rows[1][1:] == ('gbsa_running', None)
    assert rows[8][1:] == ('gbsa_running', None)
    q.poll_queue(1, args, root, prime.load_job(root))
    assert launches == list(range(1, 9))
    completed.update(range(1, 11))
    assert q.poll_queue(1, args, root, prime.load_job(root)) == 9
    assert prime.load_job(root) is None
    assert len(launches) == len(set(launches)) == 10
    assert all(row[2] == -35 for row in database.mmgbsa_submission_rows(args.db_path, range(1, 11)).values())


@pytest.mark.parametrize('legacy', [False, True])
def test_total_elapsed_survives_worker_restart(tmp_path, monkeypatch, capsys, legacy):
    monkeypatch.setattr(q.time, 'time_ns', lambda: 1_000_000_000_000)
    monkeypatch.setattr(q.time, 'time', lambda: 1000.0)
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=1)
    if legacy:
        del record['started_at']
        prime._save(root, record)
    assert q.poll_queue(1, args, root, prime.load_job(root)) is None
    assert 'total elapsed=' not in capsys.readouterr().out
    assert prime.load_job(root)['started_at'] == 1000.0

    monkeypatch.setattr(q.time, 'time', lambda: 4661.2)
    completed.add(1)
    assert q.poll_queue(1, args, root, prime.load_job(root)) == 1
    assert 'MM-GBSA queue completed; total elapsed=01:01:01.2' in capsys.readouterr().out
    saved = json.loads((root / f"{record['run_dir']}.completed.json").read_text())
    assert saved['started_at'] == 1000.0
    assert saved['finished_at'] == 4661.2
    assert launches == [1]


def test_total_elapsed_includes_terminal_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(q.time, 'time_ns', lambda: 1_000_000_000_000)
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=1)
    q.poll_queue(1, args, root, record)
    capsys.readouterr()

    def failure(command, folder, ids, **kw):
        prime.job_file(folder).unlink()
        raise prime.PrimeAtomTypingError(ids[0])

    monkeypatch.setattr(prime, 'submit_or_poll', failure)
    monkeypatch.setattr(q.time, 'time', lambda: 1012.5)
    assert q.poll_queue(1, args, root, prime.load_job(root)) == 0
    output = capsys.readouterr().out
    assert 'total elapsed=00:00:12.5' in output
    assert 'failed=1' in output


def test_unknown_status_or_launch_holds_slot_without_duplicates(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    q.poll_queue(1, args, root, record)
    submit = prime.submit_or_poll
    def uncertain(command, folder, ids, **kw):
        if ids == [1]:
            raise RuntimeError('JobServer temporarily unavailable')
        return submit(command, folder, ids, **kw)
    monkeypatch.setattr(prime, 'submit_or_poll', uncertain)
    completed.add(2)
    q.poll_queue(1, args, root, prime.load_job(root))
    assert launches == list(range(1, 9))
    assert prime.load_job(root)['entries']['1'] == 'active'
    assert prime.load_job(q._folder(root, record, 1))['job_id'] == 'job-1'


def test_terminal_failure_only_marks_one_compound_and_refills(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    q.poll_queue(1, args, root, record)
    submit = prime.submit_or_poll
    def failure(command, folder, ids, **kw):
        if ids == [3]:
            prime.job_file(folder).unlink()
            raise prime.PrimeAtomTypingError(3)
        return submit(command, folder, ids, **kw)
    monkeypatch.setattr(prime, 'submit_or_poll', failure)
    q.poll_queue(1, args, root, prime.load_job(root))
    assert launches == list(range(1, 9))
    assert database.mmgbsa_submission_rows(args.db_path, [3])[3][1] == 'failed'
    assert prime.load_job(root)['entries']['3'] == 'failed'


def test_license_retry_releases_slot_for_another_compound(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    submit = prime.submit_or_poll
    def deferred(command, folder, ids, **kw):
        if ids == [2]:
            prime._save(folder, {'job_id': None, 'launching': False, 'retry_at': 9999999999})
            return False
        return submit(command, folder, ids, **kw)
    monkeypatch.setattr(prime, 'submit_or_poll', deferred)
    q.poll_queue(1, args, root, record)
    assert launches == [1, 3, 4, 5, 6, 7, 8]
    assert prime.load_job(root)['entries']['2'] == 'pending'


def test_global_limit_and_legacy_batch_prevent_extra_jobs(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    other = tmp_path / 'iter2/mmgbsa'
    other.mkdir(parents=True)
    prime._save(other, {'mode': q.MODE, 'entries': {'101': 'active', '102': 'active'}})
    q.poll_queue(1, args, root, record)
    assert launches == list(range(1, 6))
    # A legacy batch blocks new submission; no active jobs are canceled.
    prime._save(other, {'job_id': 'legacy', 'compound_ids': [101]})
    completed.add(1)
    q.poll_queue(1, args, root, prime.load_job(root))
    assert launches == list(range(1, 6))


def test_crash_after_score_commit_recovers_without_resubmission(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=1)
    q.poll_queue(1, args, root, record)
    completed.add(1)
    original = prime._save
    def crash(directory, record):
        if directory == root and record['entries']['1'] == 'done':
            raise OSError('Checkpoint failed')
        original(directory, record)
    monkeypatch.setattr(prime, '_save', crash)
    q.poll_queue(1, args, root, prime.load_job(root))
    assert database.mmgbsa_submission_rows(args.db_path, [1])[1][2] == -35
    assert prime.load_job(root)['entries']['1'] == 'active'
    monkeypatch.setattr(prime, '_save', original)
    assert q.poll_queue(1, args, root, prime.load_job(root)) == 0
    assert launches == [1]


def test_worker_recovers_queue_before_resetting_claims(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    args.gbsa_input_count = 10
    args.target_count = 10
    monkeypatch.setattr(q, 'prepare_inputs', lambda *a: None)
    mmgbsa._run_stage(args, 1)
    assert launches == list(range(1, 8))
    assert all(row[1] == 'gbsa_running' for row in database.mmgbsa_submission_rows(args.db_path, range(1, 11)).values())


def test_missing_score_is_terminal_and_releases_only_its_slot(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    q.poll_queue(1, args, root, record)
    completed.add(4)
    finish = c.finish_iteration_mmgbsa
    monkeypatch.setattr(c, 'finish_iteration_mmgbsa',
                        lambda iteration, args, folder, ids: 0 if ids == [4] else finish(iteration, args, folder, ids))
    q.poll_queue(1, args, root, prime.load_job(root))
    assert launches == list(range(1, 9))
    assert prime.load_job(root)['entries']['4'] == 'failed'
    assert database.mmgbsa_submission_rows(args.db_path, [4])[4][1] == 'failed'


@pytest.mark.parametrize('stage,interval,expected', [('mmgbsa', 30, 5), ('mmgbsa', 2, 2), ('glide', 30, 30)])
def test_supervisor_refill_interval(stage, interval, expected, tmp_path, monkeypatch):
    from molnova import driver
    waits = []
    class Stop:
        def is_set(self):
            return bool(waits)
        def wait(self, seconds):
            waits.append(seconds)
    monkeypatch.setattr(driver, 'stream_process', lambda *a: 1)
    args = SimpleNamespace(db_path=tmp_path / 'timing.sqlite')
    driver.supervise_iteration(args, tmp_path / 'config.toml', 1,
                               [(stage, 'module')], interval, Stop(), False)
    assert waits == [expected]
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT finished_at FROM stage_timing').fetchone() == (None,)


def test_obsolete_queue_is_retained_for_external_job_reconciliation(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute('DELETE FROM compound')
    with pytest.raises(prime.PrimeReconciliationPending, match='obsolete compound queue'):
        prime.reconcile_saved_job(root, args.db_path, 1)
    assert prime.load_job(root) == record
    assert not launches


def test_committed_score_without_child_record_is_not_calculated_again(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=1)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET gbsa_score=-41,state='gbsa_done'")
    (q._folder(root, record, 1) / 'mmgbsa_input-out.maegz').write_text('receptor\nCMPID_1\n')
    assert q.poll_queue(1, args, root, record) == 0
    assert not launches
    assert database.mmgbsa_submission_rows(args.db_path, [1])[1][2] == -41


def test_refills_before_checking_remaining_active_jobs(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    q.poll_queue(1, args, root, record)
    completed.add(1)
    submit = prime.submit_or_poll
    def check_order(command, folder, ids, **kw):
        if ids == [2]:
            assert launches == list(range(1, 9)), 'Freed slot must be refilled before next status check'
        return submit(command, folder, ids, **kw)
    monkeypatch.setattr(prime, 'submit_or_poll', check_order)
    q.poll_queue(1, args, root, prime.load_job(root))
    assert launches == list(range(1, 9))


def test_glide_blocks_refill_but_existing_scores_are_still_collected(tmp_path, monkeypatch):
    from molnova import schrodinger_guard
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch)
    q.poll_queue(1, args, root, record)
    completed.add(1)
    monkeypatch.setattr(schrodinger_guard, 'active_jobs', lambda *a:
                        [{'jobId':'live-glide','jobName':'glide_constrained','status':'RUNNING'}])
    q.poll_queue(1, args, root, prime.load_job(root))
    assert launches == list(range(1, 8))
    assert database.mmgbsa_submission_rows(args.db_path, [1])[1][2] == -35
    assert sum(s == 'active' for s in prime.load_job(root)['entries'].values()) == 6
    monkeypatch.setattr(schrodinger_guard, 'active_jobs', lambda *a: [])
    q.poll_queue(1, args, root, prime.load_job(root))
    assert launches == list(range(1, 9))


def test_final_coordinates_wait_for_queue_and_exclude_failed_results(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=3)
    assert q.poll_queue(1, args, root, record) is None
    output = root / 'mmgbsa_final.maegz'
    assert not output.exists()
    completed.add(1)
    submit = prime.submit_or_poll
    def failure(command, folder, ids, **kw):
        if ids == [2]:
            prime.job_file(folder).unlink()
            raise prime.PrimeAtomTypingError(2)
        return submit(command, folder, ids, **kw)
    monkeypatch.setattr(prime, 'submit_or_poll', failure)
    assert q.poll_queue(1, args, root, prime.load_job(root)) is None
    assert not output.exists()
    completed.add(3)
    assert q.poll_queue(1, args, root, prime.load_job(root)) == 1
    assert output.read_text() == 'receptor\nCMPID_1\nreceptor\nCMPID_3\n'
    assert launches == [1, 2, 3]


def test_coordinate_merge_failure_preserves_previous_file_and_retries_without_jobs(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=1)
    q.poll_queue(1, args, root, record)
    output = root / 'mmgbsa_final.maegz'
    output.write_text('previous final coordinates')
    completed.add(1)
    merge = c.run_command
    def broken(command, cwd, **kw):
        Path(command[2]).write_text('partial coordinates')
        raise RuntimeError('structure writer failed')
    monkeypatch.setattr(c, 'run_command', broken)
    with pytest.raises(RuntimeError, match='structure writer failed'):
        q.poll_queue(1, args, root, prime.load_job(root))
    assert output.read_text() == 'previous final coordinates'
    assert not (root / 'mmgbsa_final.pending.maegz').exists()
    assert prime.load_job(root)['entries']['1'] == 'done'
    assert database.mmgbsa_submission_rows(args.db_path, [1])[1][2] == -35
    monkeypatch.setattr(c, 'run_command', merge)
    assert q.poll_queue(1, args, root, prime.load_job(root)) == 0
    assert output.read_text() == 'receptor\nCMPID_1\n'
    assert launches == [1]


def test_coordinates_include_previous_completed_queues_without_duplicates(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=2)
    previous = dict(record, run_dir='queue-1', compound_ids=[1], entries={'1': 'done'})
    folder = q._folder(root, previous, 1)
    folder.mkdir(parents=True)
    (folder / 'mmgbsa_input-out.maegz').write_text('original receptor\nCMPID_1\n')
    (root / 'queue-1.completed.json').write_text(json.dumps(previous))
    q.poll_queue(1, args, root, record)
    completed.update([1, 2])
    q.poll_queue(1, args, root, prime.load_job(root))
    assert (root / 'mmgbsa_final.maegz').read_text() == 'original receptor\nCMPID_1\nreceptor\nCMPID_2\n'


def test_missing_completed_coordinates_keep_queue_recoverable(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=1)
    record['entries']['1'] = 'done'
    prime._save(root, record)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET gbsa_score=-35,state='gbsa_done'")
    with pytest.raises(RuntimeError, match='final MM-GBSA coordinates are missing'):
        q.poll_queue(1, args, root, prime.load_job(root))
    assert prime.load_job(root) is not None
    assert not launches


def test_native_merger_preserves_receptor_and_ligand_coordinates_and_properties(tmp_path, monkeypatch):
    import runpy
    import sys
    from types import ModuleType
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=2)
    record['entries'] = {'1': 'done', '2': 'done'}
    structures = {}
    originals = []
    for cid in (1, 2):
        source = q._folder(root, record, cid) / 'mmgbsa_input-out.maegz'
        source.write_text('native output')
        receptor = SimpleNamespace(title='receptor', xyz=[(float(cid), 0, 0)], property={'native': cid})
        ligand = SimpleNamespace(title=f'CMPID_{cid}', xyz=[(0, float(cid), 0)],
                                 property={'r_psp_MMGBSA_dG_Bind': -35})
        structures[str(source)] = [receptor, ligand]
        originals.extend([receptor, ligand])
    written = []
    class Writer:
        def __init__(self, path): self.path = Path(path)
        def __enter__(self): return self
        def append(self, st): written.append(st)
        def __exit__(self, *exc): self.path.write_text('native merged structures')
    module = ModuleType('schrodinger')
    module.structure = SimpleNamespace(StructureReader=lambda path: structures[path], StructureWriter=Writer)
    monkeypatch.setitem(sys.modules, 'schrodinger', module)
    def native_run(command, cwd, **kw):
        monkeypatch.setattr(sys, 'argv', list(map(str, command[1:])))
        runpy.run_path(str(command[1]), run_name='__main__')
    monkeypatch.setattr(c, 'run_command', native_run)
    q.merge_final_coordinates(args, root, record)
    assert written == originals
    assert [st.xyz for st in written] == [[(1.0, 0, 0)], [(0, 1.0, 0)], [(2.0, 0, 0)], [(0, 2.0, 0)]]
    assert [st.title for st in written] == ['receptor', 'CMPID_1', 'receptor', 'CMPID_2']
    assert written[1].property['r_psp_MMGBSA_dG_Bind'] == -35


def test_legacy_completed_batch_publishes_native_coordinates(tmp_path, monkeypatch):
    args, root, record, launches, completed = fixture(tmp_path, monkeypatch, count=2)
    prime._save(root, {'command': ['/suite/prime_mmgbsa'], 'compound_ids': [1, 2]})
    monkeypatch.setattr(prime, 'submit_or_poll', lambda *a, **kw: True)
    monkeypatch.setattr(prime, 'persist_atomtyping_failures', lambda *a: None)
    def finish(*a):
        (root / 'mmgbsa_input-out.maegz').write_bytes(b'native coordinates and properties')
        return 2
    monkeypatch.setattr(c, 'finish_iteration_mmgbsa', finish)
    assert c.run_iteration_mmgbsa(1, args, root.parent, root.parent / 'glide') == 2
    assert (root / 'mmgbsa_final.maegz').read_bytes() == b'native coordinates and properties'
    assert prime.load_job(root) is None
