import json
import runpy
import sqlite3
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from molnova import _core, database, primg_gbsa_batch as batch, primg_gbsa_async as prime
from molnova import schrodinger_guard
from molnova.stages import primg_gbsa


def project(tmp_path, monkeypatch, count=2):
    db = tmp_path / 'project.sqlite'
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany("INSERT INTO compound(name,smiles,iteration,docking_score,state) "
                         "VALUES (?,'C',1,-9,'primg_gbsa_running')", [(str(i),) for i in range(count)])
    args = SimpleNamespace(db_path=db, output=tmp_path, schrodinger=Path('/suite'),
                           host='a:128 b:128', primg_gbsa_input_count=count, target_count=count,
                           primg_gbsa_elite_count=1, primg_gbsa_flexdist=0.0)
    root = tmp_path / 'iter1/primg_gbsa'
    root.mkdir(parents=True)
    (root / 'primg_gbsa_input.maegz').write_text('native PV')
    monkeypatch.setattr(schrodinger_guard, 'active_jobs', lambda *a: [])
    record = batch.create_batch(args, root, list(range(1, count + 1)))
    return args, root, record


def test_200_inputs_launch_once_and_restart_keeps_native_request(tmp_path, monkeypatch):
    args, root, record = project(tmp_path, monkeypatch, 200)
    launches = []
    monkeypatch.setattr(prime, '_run_attempt',
                        lambda *a: (launches.append(a[0]) or (0, 'JobId: native-parent')))
    assert batch.poll_batch(1, args, root, record) is None
    assert Path(launches[0][1]).name == 'primg_gbsa.maegz'
    assert (root / 'primg_gbsa_input.maegz').read_text() == 'native PV'
    assert launches[0][-4:] == ['-HOST', 'a:4 b:3', '-NJOBS', '200']
    assert len(prime.load_job(root)['compound_ids']) == 200
    args.host = 'different:256'
    args.primg_gbsa_flexdist = 12
    monkeypatch.setattr(prime, '_job_details', lambda *a: [
        dict(jobId='native-parent', status='RUNNING')])
    assert _core.run_iteration_primg_gbsa(1, args, root.parent, tmp_path / 'glide') is None
    assert len(launches) == 1
    assert prime.load_job(root)['command'] == launches[0]
    assert not list(root.glob('queue-*'))


@pytest.mark.parametrize('mode', ['compound_queue_v1', 'native_batch_v1', None])
def test_other_iteration_reserves_entire_budget_even_if_identity_uncertain(tmp_path, monkeypatch, mode):
    args, root, record = project(tmp_path, monkeypatch)
    other = tmp_path / 'iter0/primg_gbsa'
    other.mkdir(parents=True)
    prime._save(other, dict(mode=mode, launching=True, job_id=None))
    monkeypatch.setattr(prime, '_run_attempt', lambda *a: pytest.fail('budget bypassed'))
    assert batch.poll_batch(1, args, root, record) is None
    assert prime.load_job(root)['attempt'] == 0
    assert prime.load_job(other)['launching']


def test_new_iteration_does_not_block_refill_of_legacy_queue(tmp_path, monkeypatch):
    args, root, _ = project(tmp_path, monkeypatch)
    prime.job_file(root).unlink()
    other = tmp_path / 'iter0/primg_gbsa'
    other.mkdir(parents=True)
    prime._save(other, dict(mode='compound_queue_v1', entries={'3': 'pending'}))
    assert _core.run_iteration_primg_gbsa(1, args, root.parent, tmp_path / 'glide') is None
    assert prime.load_job(root) is None
    with pytest.raises(schrodinger_guard.SubmissionBlocked):
        batch.create_batch(args, root, [1, 2])
    assert prime.load_job(root) is None


def test_flexible_batch_uses_per_complex_receptor_and_pins_protocol(tmp_path, monkeypatch):
    args, root, _ = project(tmp_path, monkeypatch)
    prime.job_file(root).unlink()
    args.primg_gbsa_flexdist = 6
    commands = []
    def prepare(command, **kw):
        commands.append(command)
        Path(command[3]).write_text('independent native complexes')
    monkeypatch.setattr(_core, 'run_command', prepare)
    record = batch.create_batch(args, root, [1, 2])
    assert record['command'][1] == str(root / 'primg_gbsa.maegz')
    assert record['command'][-8:] == ['-ligand', 'atom.i_molnova_batch_ligand 1',
                                      '-rflexdist', '6.0', '-rflexgroup', 'side',
                                      '-out_type', 'COMPLEX']
    assert (root / 'primg_gbsa.maegz').is_file()
    assert len(commands) == 1
    assert record['primg_gbsa_flexdist'] == 6


@pytest.mark.parametrize('group', ['residue', 'side'])
def test_saved_flexible_batch_keeps_exact_protocol_on_launch_and_restart(tmp_path, monkeypatch, group):
    args, root, record = project(tmp_path, monkeypatch)
    # Existing manifests, including not-yet-launched requests, own their protocol.
    record['command'] += ['-rflexdist', '6.0', '-rflexgroup', group,
                          '-out_type', 'COMPLEX']
    record['primg_gbsa_flexdist'] = 6.0
    prime._save(root, record)
    expected = list(record['command'])
    launches = []
    monkeypatch.setattr(prime, '_run_attempt',
                        lambda *a: (launches.append(list(a[0])) or (0, 'JobId: saved-parent')))
    assert batch.poll_batch(1, args, root, prime.load_job(root)) is None
    assert launches == [expected]
    args.primg_gbsa_flexdist = 0.0
    args.host = 'changed:128'
    monkeypatch.setattr(prime, '_job_details', lambda *a: [
        dict(jobId='saved-parent', status='RUNNING')])
    assert batch.poll_batch(1, args, root, prime.load_job(root)) is None
    assert launches == [expected]
    assert prime.load_job(root)['command'] == expected


def test_failed_native_parent_keeps_streamed_result_and_retries_only_unscored(tmp_path, monkeypatch):
    args, root, record = project(tmp_path, monkeypatch)
    record.update(job_id='parent', attempt=1)
    prime._save(root, record)
    jobs = [dict(jobId='parent', status='FAILED'),
            dict(jobId='child', parentJobId='parent', jobName='primg_gbsa.00001', status='DONE')]
    monkeypatch.setattr(prime, '_job_details', lambda *a: jobs)
    def fetch(command, **kw):
        if command[1] == 'tail-file':
            kw['stdout'].write(b'native result')
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(prime.subprocess, 'run', fetch)
    monkeypatch.setattr(prime, '_current_diagnostics', lambda *a: 'insufficient licenses')
    monkeypatch.setattr(prime, '_run_attempt', lambda *a: pytest.fail('completed compounds resubmitted'))
    def extract(command, **kw):
        Path(command[-1]).write_text('id\tprimg_gbsa_score\n1\t-40\n')
    monkeypatch.setattr(_core, 'run_command', extract)
    published = []
    monkeypatch.setattr(batch, 'publish_coordinates', lambda *a: published.append(a[-1]))
    with pytest.raises(RuntimeError, match='only unscored'):
        primg_gbsa._run_stage(args, 1)
    assert prime.load_job(root) is None
    archived = json.loads(next(root.glob('batch-*.failed.json')).read_text())
    assert archived['job_id'] == 'parent'
    assert archived['imported_subjobs'] == ['child']
    assert published == [[1, 2]]
    assert database.primg_gbsa_submission_rows(args.db_path, [1, 2]) == {
        1: (1, 'primg_gbsa_done', -40), 2: (1, 'docked', None)}
    assert database.claim_primg_gbsa_candidates(args.db_path, 1, 2, 2) == [2]


def test_final_coordinate_failure_keeps_job_and_retries_without_submission(tmp_path, monkeypatch):
    args, root, record = project(tmp_path, monkeypatch)
    record.update(job_id='parent', attempt=1)
    prime._save(root, record)
    monkeypatch.setattr(prime, '_job_details', lambda *a: [dict(jobId='parent', status='DONE')])
    (root / 'primg_gbsa_input-out.maegz').write_text('native completed batch')
    monkeypatch.setattr(prime.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(prime, '_run_attempt', lambda *a: pytest.fail('duplicate calculation'))
    def finish(iteration, args, directory, ids):
        scores = directory / 'scores.tsv'
        scores.write_text('id\tprimg_gbsa_score\n1\t-40\n2\t-30\n')
        return _core.update_primg_gbsa_scores(scores, args.db_path, allowed_ids=ids)
    monkeypatch.setattr(_core, 'finish_iteration_primg_gbsa', finish)
    def fail(*a):
        raise OSError('merge interrupted')
    monkeypatch.setattr(batch, 'publish_coordinates', fail)
    with pytest.raises(OSError):
        batch.poll_batch(1, args, root, record)
    assert prime.load_job(root)['job_id'] == 'parent'
    assert database.primg_gbsa_progress_counts(args.db_path, 1) == (0, 2)
    monkeypatch.setattr(batch, 'publish_coordinates', lambda *a: None)
    assert batch.poll_batch(1, args, root, prime.load_job(root)) == 0
    assert prime.load_job(root) is None
    assert next(root.glob('batch-*.completed.json')).is_file()


def test_failed_parent_does_not_retire_when_completed_child_output_is_unavailable(tmp_path, monkeypatch):
    args, root, record = project(tmp_path, monkeypatch)
    record.update(job_id='parent', attempt=1)
    prime._save(root, record)
    monkeypatch.setattr(prime, '_job_details', lambda *a: [
        dict(jobId='parent', status='FAILED'),
        dict(jobId='child', parentJobId='parent', status='DONE', jobName='primg_gbsa.00001')])
    monkeypatch.setattr(prime.subprocess, 'run',
                        lambda *a, **kw: SimpleNamespace(returncode=1, stderr=b'unavailable'))
    monkeypatch.setattr(prime, '_run_attempt', lambda *a: pytest.fail('duplicate submission'))
    with pytest.raises(prime.PrimeReconciliationPending, match='still require collection'):
        primg_gbsa._run_stage(args, 1)
    assert prime.load_job(root)['job_id'] == 'parent'
    assert not list(root.glob('batch-*.failed.json'))
    assert database.primg_gbsa_progress_counts(args.db_path, 1) == (2, 0)


def test_coordinates_keep_previous_scored_pose_deduplicate_and_put_receptor_first(tmp_path, monkeypatch):
    args, root, _ = project(tmp_path, monkeypatch)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute('UPDATE compound SET primg_gbsa_score=-40')
    previous = root / 'previous-1/primg_gbsa_final.maegz'
    child = root / 'subjobs/parent/child/result.maegz'
    combined = root / 'primg_gbsa_input-out.maegz'
    def st(cid=None, coordinates=None):
        return SimpleNamespace(title=f'CMPID_{cid}' if cid else 'receptor', property={},
                               coordinates=coordinates)
    existing, new = st(1, 'original optimized'), st(2, 'new optimized')
    receptor = st(coordinates='frozen receptor')
    data = {str(previous): [existing], str(child): [new],
            str(combined): [receptor, st(1, 'recalculated'), new, st(99, 'unsubmitted')]}
    for path in (previous, child, combined):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    written = []
    class Writer:
        def __init__(self, output):
            self.output = Path(output)
        def __enter__(self):
            return self
        def append(self, item):
            written.append(item)
        def __exit__(self, *exc):
            self.output.write_text('native structures')
    module = ModuleType('schrodinger')
    module.structure = SimpleNamespace(StructureReader=lambda filename: iter(data[str(filename)]),
                                       StructureWriter=Writer)
    monkeypatch.setitem(sys.modules, 'schrodinger', module)
    def run(command, **kw):
        monkeypatch.setattr(sys, 'argv', list(map(str, command[1:])))
        runpy.run_path(str(command[1]), run_name='__main__')
    monkeypatch.setattr(_core, 'run_command', run)
    batch.publish_coordinates(args, root, [1, 2])
    assert written == [receptor, existing, new]
    assert (root / 'primg_gbsa_final.maegz').read_text() == 'native structures'
    assert not (root / 'primg_gbsa_final.pending.maegz').exists()
