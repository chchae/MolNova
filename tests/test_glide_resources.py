import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database
from molnova.config import glide_group_resources, schrodinger_job_options
from molnova.glide_recovery import save_record, load_record
from molnova.stages import glide


def requested_slots(host):
    return sum(int(entry.rsplit(':', 1)[1]) for entry in host.split())


def test_weighted_allocation_caps_small_groups_and_uses_budget():
    counts = {1: 3, 2: 100, 3: 900}
    result = glide_group_resources('compute:124', None, counts)
    slots = {ref: requested_slots(host) for ref, host in result.items()}
    assert sum(slots.values()) == 124
    assert 1 <= slots[1] <= 3
    assert slots[1] < slots[2] < slots[3]
    assert glide_group_resources('compute:128', 124, counts) == result


@pytest.mark.parametrize('budget', [1, 3, 5, 12, 124, 2000])
def test_allocation_never_exceeds_budget_or_structure_counts(budget):
    counts = {1: 1, 2: 3, 3: 17, 4: 350, 5: 200}
    result = glide_group_resources(f'compute:{budget}', None, counts)
    assert sum(requested_slots(host) for host in result.values()) == min(budget, sum(counts.values()))
    assert all(1 <= requested_slots(host) <= counts[ref] for ref, host in result.items())


def test_equal_groups_split_cores_evenly_with_deterministic_remainders():
    result = glide_group_resources('compute:124', None, {i: 100 for i in range(23)})
    sizes = [requested_slots(host) for host in result.values()]
    assert sizes.count(6) == 9
    assert sizes.count(5) == 14
    assert result == glide_group_resources('compute:124', None, {i: 100 for i in reversed(range(23))})


def test_multiple_hosts_respect_each_hosts_remaining_slots():
    result = glide_group_resources('a:4 b:8', None, {1: 20, 2: 60}, ['a:2 b:3'])
    totals = {'a': 0, 'b': 0}
    for host in result.values():
        for entry in host.split():
            name, value = entry.split(':')
            totals[name] += int(value)
    assert totals == {'a': 2, 'b': 5}


@pytest.mark.parametrize('reserved', [[None], ['compute'], ['other:2'], ['compute:128']])
def test_uncertain_or_oversized_legacy_requests_block_only_new_submissions(reserved):
    assert glide_group_resources('compute:124', None, {1: 100}, reserved) == {}


def test_bare_host_preserves_vendor_defaults_and_empty_groups_do_not_launch():
    assert glide_group_resources('compute', None, {1: 100, 2: 20}) == {1: 'compute', 2: 'compute'}
    assert glide_group_resources('compute:124', None, {}) == {}


@pytest.mark.parametrize('bad', [0, -1, True, 2.5, '10'])
def test_invalid_counts_are_rejected(bad):
    with pytest.raises(ValueError, match='ligand counts'):
        glide_group_resources('compute:124', None, {1: bad})


@pytest.mark.parametrize('legacy', [None, 'compute:5', 'compute:124', 'unknown'])
def test_stage_uses_prepared_counts_reserves_live_jobs_and_keeps_native_commands(
        tmp_path, monkeypatch, capsys, legacy):
    db = tmp_path / 'project.sqlite'
    with database.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        for ref in (1, 2, 3):
            conn.execute("INSERT INTO compound(id,name,smiles,iteration,state) VALUES (?,?,'C',0,'reference')",
                         (ref, f'REF{ref}'))
            conn.execute("INSERT INTO compound(id,name,smiles,iteration,state,similar_to) "
                         "VALUES (?,?,'C',1,'ligprepped',?)", (ref + 3, f'GEN{ref}', ref))
    args = SimpleNamespace(db_path=db, output=tmp_path, schrodinger=Path('/suite'),
                           host='compute:128', glide_cpus=12, mcs_smarts=None,
                           reference_poses=Path('refs.maegz'), grid=Path('grid.zip'))
    prepared = tmp_path / 'iter1/ligprep/ligprep.maegz'
    prepared.parent.mkdir(parents=True)
    prepared.write_text('input')
    root = tmp_path / 'iter1/glide'
    root.mkdir()
    glide.prepare_glide_files(root, prepared)
    if legacy:
        directory = root / 'ref_1'
        directory.mkdir()
        command = ['/suite/glide', 'input.in', '-HOST', legacy, '-NJOBS', '5']
        if legacy == 'unknown':
            command = []
        save_record(directory, dict(job_id='existing', command=command))
        original_record = load_record(directory)
    submitted = {}
    def split(*a):
        # One original compound each, but very different LigPrep state counts.
        (root / 'ligprep_counts.json').write_text(json.dumps({1: 2, 2: 10, 3: 100}))
        files = {}
        for ref in (1, 2, 3):
            directory = root / f'ref_{ref}'
            directory.mkdir(exist_ok=True)
            files[ref] = directory / 'ligprep_group.maegz'
            files[ref].write_text('ligands')
        return files
    def status(suite, directory, *a, **k):
        ref = int(directory.name.split('_')[1])
        return 'RUNNING' if (ref == 1 and legacy) or ref in submitted else None
    def submit(suite, inp, directory, host, **kwargs):
        ref = int(directory.name.split('_')[1])
        assert not (legacy and ref == 1)
        assert kwargs.get('cpus') is None
        submitted[ref] = host
        options = schrodinger_job_options(host, split_jobs=True)
        assert int(options[options.index('-NJOBS') + 1]) == requested_slots(host)
        return {}
    monkeypatch.setattr(_core, 'assign_reference_compounds', lambda *a: None)
    monkeypatch.setattr(_core, 'split_ligprep_by_reference', split)
    monkeypatch.setattr(_core, 'extract_reference_pose', lambda *a: None)
    monkeypatch.setattr(_core, 'submit_glide', submit)
    monkeypatch.setattr(glide, 'active_jobs', lambda *a: [])
    monkeypatch.setattr(glide, 'group_status', status)
    monkeypatch.setattr(glide.time, 'monotonic', itertools.count().__next__)
    monkeypatch.setattr(glide.time, 'sleep', lambda *a: None)
    glide._run_stage(args, SimpleNamespace(iteration=1, poll_interval=1,
                                           completion_fraction=0, tail_timeout=0))
    if legacy in ('unknown', 'compute:124'):
        assert submitted == {}
    else:
        assert requested_slots(submitted[3]) > requested_slots(submitted[2])
        assert sum(requested_slots(host) for host in submitted.values()) == (7 if legacy else 12)
        assert '100 prepared ligands; NJOBS=' in capsys.readouterr().out
    if legacy:
        assert load_record(root / 'ref_1') == original_record
    with database.connect(db) as conn:
        assert conn.execute('SELECT count(*) FROM compound WHERE iteration=1 AND state=\'glide_running\'').fetchone()[0] == 3


def test_splitter_records_actual_prepared_structure_counts(tmp_path, monkeypatch):
    import runpy
    import sys
    import types
    class Reader:
        def __init__(self, path):
            pass
        def __iter__(self):
            return iter(SimpleNamespace(title=title) for title in
                        ['CMPID_10', 'CMPID_10', 'CMPID_20', 'CMPID_999', 'unmapped'])
    class Writer:
        def __init__(self, path):
            self.path = Path(path)
        def append(self, st):
            with self.path.open('a') as f:
                f.write(st.title + '\n')
        def close(self):
            pass
    fake = types.ModuleType('schrodinger')
    fake.structure = SimpleNamespace(StructureReader=Reader, StructureWriter=Writer)
    monkeypatch.setitem(sys.modules, 'schrodinger', fake)
    map_file = _core.write_reference_map({1: [(10, 'A', 'C', 1)], 2: [(20, 'B', 'CC', 1)]}, tmp_path)
    script = _core.create_ligprep_splitter(tmp_path)
    monkeypatch.setattr(sys, 'argv', [str(script), 'input.maegz', str(map_file), str(tmp_path)])
    runpy.run_path(str(script), run_name='__main__')
    assert json.loads((tmp_path / 'ligprep_counts.json').read_text()) == {'1': 2, '2': 1}
    assert (tmp_path / 'ref_1/ligprep_group.maegz').read_text().splitlines() == ['CMPID_10', 'CMPID_10']
