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
        return updated
    monkeypatch.setattr(prime, 'submit_or_poll', submit)
    monkeypatch.setattr(c, 'finish_iteration_mmgbsa', finish)
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
    monkeypatch.setattr(driver, 'stream_process', lambda *a: 0)
    driver.worker_loop(stage, 'module', tmp_path / 'config.toml', tmp_path, interval, Stop(), False)
    assert waits == [expected]


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
