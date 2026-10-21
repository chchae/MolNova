import concurrent.futures
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from molnova import _core, amber_gbsa, amber_runner, cli, database, driver
from molnova.config import amber_settings
from molnova.stages import amber_gbsa as stage, generate, primg_gbsa
from molnova.states import CompoundState as State


@pytest.fixture
def project(tmp_path):
    db = tmp_path / 'project.sqlite'
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany(
            'INSERT INTO compound(name,smiles,iteration,docking_score,state) VALUES (?, ?, 1, ?, ?)',
            [(f'GEN_I01_{i:06d}', 'CCO', -i, State.DOCKED) for i in range(1, 5)])
    receptor = tmp_path / 'receptor.maegz'
    receptor.touch()
    return SimpleNamespace(db_path=db, output=tmp_path, target_count=4, primg_gbsa_input_count=2,
                           amber_gbsa_enabled=True, amber_keep_count=3, amber_cpus=128,
                           amber_receptor=receptor, amber_charge_method='bcc',
                           amber_remote={'host':'tensor', 'remote_env':'/env', 'conda_init':'/conda.sh',
                                         'remote_work_dir':'/runs'}, primg_gbsa_elite_count=1,
                           max_iteration=3, synthetic_feasibility_enabled=False)


def scores(project):
    ids = database.claim_amber_candidates(project.db_path, 1, 4)
    database.store_amber_results(project.db_path, 1,
        [{'id': cid, 'score': -10 * cid} for cid in ids], ids)


def test_non_destructive_amber_migration(project):
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state='primg_gbsa_running',prime_gbsa_score=-31 WHERE id=1")
        conn.execute('ALTER TABLE compound DROP COLUMN amber_gbsa_score')
        _core.ensure_schema_columns(conn)
        _core.ensure_schema_columns(conn)
        assert conn.execute('SELECT state,prime_gbsa_score,amber_gbsa_score FROM compound WHERE id=1').fetchone() == ('primg_gbsa_running',-31,None)


def test_atomic_amber_claims_do_not_overlap(project):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: database.claim_amber_candidates(project.db_path, 1, 4), range(2)))
    assert sorted(len(r) for r in results) == [0, 4]
    assert not set(results[0]) & set(results[1])


def test_all_amber_work_precedes_prime_and_generation(project):
    assert 'Amber' in database.iteration_completion_reason(project.db_path, 1, 4, 2, 1, 3)
    assert database.claim_primg_gbsa_candidates(project.db_path, 1, 2, 4, 3) == []
    ids = database.claim_amber_candidates(project.db_path, 1, 4)
    database.store_amber_results(project.db_path, 1, [{'id':4,'score':-80}], ids)
    assert database.claim_primg_gbsa_candidates(project.db_path, 1, 2, 4, 3) == []
    database.store_amber_results(project.db_path, 1,
        [{'id':i,'score':-i*10} for i in (1,2,3)], ids)
    assert database.claim_primg_gbsa_candidates(project.db_path, 1, 2, 4, 3) == [4,3]


def test_prime_uses_amber_order_keeps_scored_positions_and_ties(project):
    scores(project)
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET prime_gbsa_score=-50,state='primg_gbsa_done' WHERE id=4")
        conn.execute('UPDATE compound SET amber_gbsa_score=-30,docking_score=-3 WHERE id=2')
    assert database.primg_gbsa_candidates(project.db_path, 1, 2, 4, 3) == [2]
    assert [r[0] for r in database.prime_selection_rows(project.db_path,1,2,3)] == [4,2]
    # Legacy disabled projects retain Glide ranking.
    assert database.primg_gbsa_candidates(project.db_path, 1, 2, 4) == [2]


def test_1000_to_500_to_200(project):
    with database.connect(project.db_path) as conn:
        conn.execute('DELETE FROM compound')
        conn.executemany(
            "INSERT INTO compound(id,name,smiles,iteration,state,docking_score,amber_gbsa_score) "
            "VALUES (?,?,'C',1,'docked',?,?)",
            [(i,f'GEN_I01_{i:06d}',-i,-(1001-i)) for i in range(1,1001)])
    assert database.claim_primg_gbsa_candidates(project.db_path,1,200,1000,500) == list(range(501,701))


@pytest.mark.parametrize('bad', [{'id':99,'score':-5}, {'id':2,'score':float('inf')},
                                  {'id':2,'error':'x','score':-8}, {'id':True,'score':-5},
                                  {'id':2,'score':True}, {'id':2,'error':''}])
def test_invalid_batch_rolls_back_all_results(project, bad):
    ids=database.claim_amber_candidates(project.db_path,1,4)
    with pytest.raises(ValueError):
        database.store_amber_results(project.db_path,1,[{'id':1,'score':-20},bad],ids)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT COUNT(amber_gbsa_score) FROM compound').fetchone()[0] == 0


def test_import_preserves_prime_fep_and_replays_idempotently(project):
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state='fep_done',prime_gbsa_score=-40,fep_score=-9 WHERE id=1")
    ids=database.claim_amber_candidates(project.db_path,1,4)
    results=[{'id':1,'score':-25},{'id':2,'error':'parameterization failed'}]
    assert database.store_amber_results(project.db_path,1,results,ids) == 1
    assert database.store_amber_results(project.db_path,1,results,ids) == 0
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT state,amber_gbsa_score,prime_gbsa_score,fep_score FROM compound WHERE id=1').fetchone() == ('fep_done',-25,-40,-9)
        assert conn.execute('SELECT state,failed_stage,failure_message FROM compound WHERE id=2').fetchone() == ('failed','amber_gbsa','parameterization failed')


def test_amber_failures_are_terminal_but_cannot_rank(project):
    ids=database.claim_amber_candidates(project.db_path,1,4)
    database.store_amber_results(project.db_path,1,
        [{'id':cid,'error':'unsupported atom'} if cid==4 else {'id':cid,'score':-cid*10} for cid in ids],ids)
    assert database.amber_completion_reason(project.db_path,1,4) is None
    assert database.primg_gbsa_candidates(project.db_path,1,2,4,3) == [3,2]


def test_orphan_claim_recovery_restores_scored_state(project):
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET prime_gbsa_score=-40,state='primg_gbsa_done' WHERE id=1")
    database.claim_amber_candidates(project.db_path,1,4)
    database.reset_unsubmitted_amber(project.db_path,1)
    with database.connect(project.db_path) as conn:
        assert conn.execute('SELECT state FROM compound WHERE id=1').fetchone()[0] == 'primg_gbsa_done'


def make_record(project):
    directory=project.output/'iter1/amber_gbsa'
    directory.mkdir(parents=True)
    ids=database.claim_amber_candidates(project.db_path,1,4)
    record=amber_gbsa.new_record(project,1,directory,ids)
    record['prepared']=True
    folder=directory/record['run_id']; folder.mkdir()
    (folder/'input.json').write_text(json.dumps({'identities':database.amber_identity_rows(project.db_path,1,ids)}))
    amber_gbsa.save(directory/'amber_job.json',record)
    return directory,record


def test_postcommit_recovery_requires_remote_terminal_and_lock_release(project, monkeypatch):
    directory,record=make_record(project)
    snapshot={'status':'done','active':True,'results':[{'id':i,'score':-i} for i in range(1,5)]}
    monkeypatch.setattr(amber_gbsa,'ssh',lambda *a,**kw: SimpleNamespace(stdout=json.dumps(snapshot)))
    monkeypatch.setattr(amber_gbsa,'launch',lambda *a:pytest.fail('Duplicate launch'))
    monkeypatch.setattr(amber_gbsa,'download_logs',lambda *a:None)
    assert not amber_gbsa.poll(project,1,directory,record)
    assert (directory/'amber_job.json').exists()
    snapshot['active']=False
    assert amber_gbsa.poll(project,1,directory,record)
    assert not (directory/'amber_job.json').exists()
    assert (directory/(record['run_id']+'.completed.json')).exists()


def test_driver_recovers_saved_amber_even_with_complete_scores(project):
    make_record(project)
    database.store_amber_results(project.db_path,1,[{'id':i,'score':-i} for i in range(1,5)],range(1,5))
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET state='primg_gbsa_done',prime_gbsa_score=-40")
    assert driver.choose_iteration(project) == 1


def test_standalone_amber_waits_for_uncertain_glide_identity(project, capsys):
    directory=project.output/'iter1/glide/ref_1';directory.mkdir(parents=True)
    (directory/'glide_job.json').write_text(json.dumps({'job_id':None}))
    stage._run_stage(project,1)
    assert 'Glide submission identity uncertain' in capsys.readouterr().out
    with database.connect(project.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM compound WHERE state='amber_running'").fetchone()[0] == 0


def test_changed_pose_fingerprint_blocks_stale_amber_scores(project):
    poses=project.output/'iter1/glide/best_poses.maegz';poses.parent.mkdir(parents=True)
    poses.write_bytes(b'old poses')
    record={'source_fingerprints':amber_gbsa.source_fingerprints(project,1)}
    poses.write_bytes(b'new poses')
    with pytest.raises(RuntimeError,match='changed'):
        amber_gbsa.verify_sources(project,1,record)


def test_disconnected_ssh_keeps_claims_and_submission(project,monkeypatch):
    directory,record=make_record(project)
    monkeypatch.setattr(amber_gbsa,'ssh',lambda *a,**kw: (_ for _ in ()).throw(OSError('network')))
    with pytest.raises(OSError):
        amber_gbsa.poll(project,1,directory,record)
    assert (directory/'amber_job.json').exists()
    with database.connect(project.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM compound WHERE state='amber_running'").fetchone()[0]==4


def test_saved_run_rejects_changed_identity(project,monkeypatch):
    directory,record=make_record(project)
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET smiles='CC' WHERE id=1")
    monkeypatch.setattr(amber_gbsa,'ssh',lambda *a,**kw:pytest.fail('Remote access with wrong identity'))
    with pytest.raises(RuntimeError,match='identity'):
        amber_gbsa.poll(project,1,directory,record)


def test_direct_prime_and_generation_cannot_bypass_saved_amber_run(project,monkeypatch,capsys):
    directory,record=make_record(project)
    database.store_amber_results(project.db_path,1,[{'id':cid,'score':-cid} for cid in range(1,5)],range(1,5))
    monkeypatch.setattr(primg_gbsa,'reconcile_saved_job',lambda *a:None)
    monkeypatch.setattr(_core,'run_iteration_primg_gbsa',lambda **kw:pytest.fail('Prime started'))
    primg_gbsa._run_stage(project,1)
    assert 'Amber remote work' in capsys.readouterr().out
    generate._run_stage(project,2)
    assert 'Amber work still requires recovery' in capsys.readouterr().out


def test_rankings_link_db_names_and_cmpid(project):
    scores(project)
    directory=project.output/'iter1/amber_gbsa';directory.mkdir(parents=True)
    amber_gbsa.write_rankings(project,1,directory)
    assert len((directory/'amber_top.tsv').read_text().splitlines())==4
    lines=(directory/'prime_candidates.tsv').read_text().splitlines()
    assert len(lines)==3 and 'CMPID_4\tGEN_I01_000004' in lines[1]


def test_environment_activates_conda_and_limits_native_threads():
    command=amber_gbsa.environment_command({'conda_init':'/a conda.sh','remote_env':'/remote/env'},'MMPBSA.py --version')
    assert 'conda activate' in command and 'CONDA_PREFIX/amber.sh' in command
    assert 'OMP_NUM_THREADS=1' in command and 'OPENBLAS_NUM_THREADS=1' in command


@pytest.mark.parametrize('key,value',[('amber-cpus',True),('amber-cpus',0),('amber-keep-count',100),
                                     ('amber-remote-env','env'),('amber-ssh-host','-bad')])
def test_config_validation(tmp_path,key,value):
    config={'out-dir':'output','amber-gbsa-enabled':True,'prime-gbsa-input-count':200,key:value}
    with pytest.raises(ValueError):amber_settings(config,tmp_path)


def test_config_paths_and_stage_order(tmp_path):
    settings=amber_settings({'out-dir':'output'},tmp_path)
    assert settings['amber_receptor']==tmp_path/'output/iter00/08-primg_gbsa/receptor.maegz'
    assert settings['amber_remote']['remote_env']=='/home1/amber26/envs/AmberTools26'
    assert [s for s,_ in driver.build_stages(False,False,True)]==['generate','druglike','ligprep','glide','amber-gbsa','primg_gbsa']


def test_cli_dispatch(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(stage,'main',lambda args:calls.append(args))
    cli.main(['amber-gbsa',str(tmp_path/'project.toml'),'--iteration','1'])
    assert calls==[[str(tmp_path/'project.toml'),'--iteration','1']]


def test_score_parser_rejects_missing_multiple_and_nonfinite(tmp_path):
    path=tmp_path/'out.dat'
    for value in ('', 'DELTA TOTAL nan', 'DELTA TOTAL -20\nDELTA TOTAL -21'):
        path.write_text(value)
        with pytest.raises(ValueError):amber_runner.parse_score(path)
    path.write_text('DELTA TOTAL   -20.15  0.0  0.0\n')
    assert amber_runner.parse_score(path)==-20.15


def test_charge_parameterization_coordinates_are_restored_and_identity_checked(tmp_path):
    original=tmp_path/'input.mol2';typed=tmp_path/'ligand.mol2'
    source='@<TRIPOS>ATOM\n1 C1 1.0 2.0 3.0 C.3 1 LIG 0\n2 O2 4.0 5.0 6.0 O.3 1 LIG 0\n@<TRIPOS>BOND\n1 1 2 1\n'
    original.write_text(source)
    typed.write_text(source.replace('1.0 2.0 3.0','7.0 8.0 9.0').replace('C.3','c3'))
    amber_runner.restore_pose(original,typed)
    assert '1 C1 1.0 2.0 3.0 c3' in typed.read_text()
    typed.write_text(source.replace('C1','C9'))
    with pytest.raises(ValueError,match='identity'):
        amber_runner.restore_pose(original,typed)


def test_native_child_keeps_recovery_lock_after_coordinator_exit(tmp_path):
    marker=tmp_path/'ready'
    script=r'''
import fcntl,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);lock=(root/'run.lock').open('a')
fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
subprocess.Popen([sys.executable,'-c','import time;time.sleep(1.5)'],pass_fds=(lock.fileno(),),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
(root/'ready').touch()
os._exit(0)
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True)
    assert marker.exists()
    command=[sys.executable,str(Path(amber_runner.__file__)),'collect',str(tmp_path)]
    assert json.loads(subprocess.check_output(command))['active']
    time.sleep(1.6)
    assert not json.loads(subprocess.check_output(command))['active']


@pytest.fixture
def references(project):
    with database.connect(project.db_path) as conn:
        conn.execute("UPDATE compound SET iteration=0,state='reference',docking_score=NULL")
        conn.execute("UPDATE compound SET state='primg_gbsa_done',prime_gbsa_score=-40 WHERE id=4")
    project.reference_poses = project.output / 'references.maegz'
    project.reference_poses.write_bytes(b'reference poses')
    return project


def test_reference_amber_precedes_prime_without_counts_or_docking(references):
    args = references
    assert database.primg_gbsa_candidates(args.db_path, 0, 1, 1000, 1) == []
    assert database.claim_amber_candidates(args.db_path, 0, 1000) == [1, 2, 3, 4]
    assert database.claim_amber_candidates(args.db_path, 0, 1000) == []
    database.store_amber_results(args.db_path, 0, [{'id':1,'score':-10}], range(1,5))
    assert database.claim_primg_gbsa_candidates(args.db_path, 0, 1, 1000, 1) == []
    database.store_amber_results(args.db_path, 0,
        [{'id':i,'score':-i} for i in (2,3,4)], range(1,5))
    assert database.amber_completion_reason(args.db_path, 0, 1000) is None
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT state,prime_gbsa_score FROM compound ORDER BY id').fetchall() == [
            ('reference',None), ('reference',None), ('reference',None), ('primg_gbsa_done',-40)]
    assert database.claim_primg_gbsa_candidates(args.db_path, 0, 1, 1000, 1) == [1,2,3]


def test_reference_orphan_claims_and_terminal_failures(references):
    args = references
    database.claim_amber_candidates(args.db_path, 0, 1000)
    database.reset_unsubmitted_amber(args.db_path, 0)
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT state FROM compound ORDER BY id').fetchall() == [
            ('reference',), ('reference',), ('reference',), ('primg_gbsa_done',)]
    ids = database.claim_amber_candidates(args.db_path, 0, 1000)
    database.store_amber_results(args.db_path, 0,
        [{'id':i,'error':'unsupported atom'} for i in ids], ids)
    assert database.amber_completion_reason(args.db_path, 0, 1000) is None
    assert database.reference_primg_gbsa_completion_reason(args.db_path) is None
    assert database.claim_amber_candidates(args.db_path, 0, 1000) == []


def test_reference_driver_selects_missing_amber_and_recovers_prime_first(references):
    args = references
    with database.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET prime_gbsa_score=-40,state='primg_gbsa_done'")
    assert driver.choose_iteration(args) == 0
    assert [s for s,_ in driver.reference_stages(args)] == ['amber-gbsa','primg_gbsa']
    directory = args.output / 'iter0/primg_gbsa'
    directory.mkdir(parents=True)
    (directory / 'primg_gbsa_job.json').write_text(json.dumps({'compound_ids':[1]}))
    assert [s for s,_ in driver.reference_stages(args)] == ['primg_gbsa-recovery','amber-gbsa','primg_gbsa']


@pytest.mark.parametrize('enabled', [False, True])
def test_reference_prime_waits_for_saved_amber_after_score_commit(references, monkeypatch, capsys, enabled):
    args = references
    ids = database.claim_amber_candidates(args.db_path, 0, 1000)
    directory = args.output / 'iter0/amber_gbsa'
    directory.mkdir(parents=True)
    amber_gbsa.new_record(args, 0, directory, ids)
    database.store_amber_results(args.db_path, 0, [{'id':i,'score':-i} for i in ids], ids)
    args.amber_gbsa_enabled = enabled
    monkeypatch.setattr(primg_gbsa, 'reconcile_saved_job', lambda *a:None)
    monkeypatch.setattr(_core, 'run_iteration_primg_gbsa', lambda **kw:pytest.fail('Prime started'))
    primg_gbsa._run_stage(args, 0)
    assert 'Amber remote work' in capsys.readouterr().out
    assert driver.choose_iteration(args) == 0


def test_reference_worker_and_rankings_use_all_references(references, monkeypatch):
    args = references
    args.target_count = 1000
    args.amber_keep_count = args.primg_gbsa_input_count = 1
    def poll(args, iteration, directory, record):
        assert iteration == 0 and record['compound_ids'] == [1,2,3,4]
        database.store_amber_results(args.db_path, 0,
            [{'id':i,'score':-i} for i in record['compound_ids']], record['compound_ids'])
        (directory / 'amber_job.json').unlink()
        return True
    monkeypatch.setattr(amber_gbsa, 'poll', poll)
    stage._run_stage(args, 0)
    assert len((args.output / 'iter00/07-amber_gbsa/prime_candidates.tsv').read_text().splitlines()) == 5
    assert str(args.reference_poses.resolve()) in amber_gbsa.source_fingerprints(args, 0)


def test_reference_receptor_bootstrap_uses_existing_grid_extractor(references, monkeypatch):
    args = references
    args.amber_receptor = args.output / 'iter00/08-primg_gbsa/receptor.maegz'
    args.schrodinger = Path('/suite')
    args.grid = args.output / 'grid.zip'
    calls = []
    def run(command, cwd):
        calls.append(command)
        Path(command[-1]).write_bytes(b'prepared receptor')
    monkeypatch.setattr(_core, 'run_command', run)
    amber_gbsa.prepare_reference_receptor(args)
    assert args.amber_receptor.read_bytes() == b'prepared receptor'
    assert calls[0][2] == args.grid
    assert calls[0][1].name == '_extract_receptor_from_grid.py'
    amber_gbsa.prepare_reference_receptor(args)
    assert len(calls) == 1


@pytest.mark.parametrize('duplicate', [False, True])
def test_reference_export_maps_titles_and_preserves_source(references, monkeypatch, duplicate):
    import runpy
    args = references
    args.schrodinger = Path('/suite')
    folder = args.output / 'export'
    folder.mkdir()
    receptor = SimpleNamespace(residue=[], bond=[], write=lambda path:Path(path).write_text(
        'ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n'))
    atom = SimpleNamespace(formal_charge=0, element='C', index=1)
    class Ligand:
        def __init__(self, title):
            self.title = title
            self.atom = [atom]
        def write(self, path):
            assert self.title.startswith('CMPID_')
            Path(path).write_text('@<TRIPOS>ATOM\n1 C1 1 2 3 C.3 1 LIG 0\n@<TRIPOS>BOND\n')
    original = [Ligand('GEN_I01_000001'), Ligand('GEN_I01_000002')]
    if duplicate:
        original.append(Ligand('GEN_I01_000001'))
    def reader(path):
        return [receptor] if Path(path) == args.amber_receptor else [Ligand(ligand.title) for ligand in original]
    monkeypatch.setitem(sys.modules, 'schrodinger', SimpleNamespace(structure=SimpleNamespace(StructureReader=reader)))
    def run(command, cwd):
        assert command[4] == args.reference_poses
        monkeypatch.setattr(sys, 'argv', [str(command[1]), *map(str, command[2:])])
        runpy.run_path(str(command[1]), run_name='__main__')
    monkeypatch.setattr(_core, 'run_command', run)
    if duplicate:
        with pytest.raises(ValueError, match='Multiple input poses'):
            amber_gbsa.export_inputs(args, 0, folder, [1,2])
    else:
        assert amber_gbsa.export_inputs(args, 0, folder, [1,2]) == [{'id':1,'charge':0},{'id':2,'charge':0}]
        assert (folder / 'CMPID_1/input.mol2').exists()
        assert [ligand.title for ligand in original] == ['GEN_I01_000001','GEN_I01_000002']
        assert args.reference_poses.read_bytes() == b'reference poses'


def test_new_remote_run_bundles_normalizer_and_records_version(references, monkeypatch):
    import tarfile
    from molnova.receptor_pdb import NORMALIZATION_VERSION
    args = references
    directory = args.output / 'iter0/amber_gbsa'
    directory.mkdir(parents=True)
    record = amber_gbsa.new_record(args, 0, directory, [1])
    def export(args, iteration, folder, ids):
        (folder / 'receptor.pdb').write_text('raw PDB')
        (folder / 'receptor.normalized.pdb').write_text('normalized PDB')
        (folder / 'receptor_bonds.in').write_text('')
        ligand = folder / 'CMPID_1'
        ligand.mkdir()
        (ligand / 'input.mol2').write_text('MOL2')
        return [{'id':1, 'charge':0}]
    def ssh(settings, command, **kwargs):
        if 'stdin' in kwargs:
            with tarfile.open(fileobj=kwargs['stdin']) as archive:
                assert {'receptor.pdb','receptor.normalized.pdb','receptor_pdb.py','worker.py','input.json'} <= set(archive.getnames())
                config = json.load(archive.extractfile('input.json'))
                assert config['receptor_normalization_version'] == NORMALIZATION_VERSION
                assert archive.extractfile('receptor_pdb.py').read() == Path(amber_gbsa.__file__).with_name('receptor_pdb.py').read_bytes()
    monkeypatch.setattr(amber_gbsa, 'export_inputs', export)
    monkeypatch.setattr(amber_gbsa, 'ssh', ssh)
    amber_gbsa.prepare_remote(args, 0, directory, record)
    assert record['prepared'] and record['input_sha256']
