"""Numbered outputs must remain discoverable by workers and restart guards."""
import json
from types import SimpleNamespace

import pytest

from molnova import amber_gbsa, config, driver, gbsa_batch, gbsa_queue
from molnova.paths import (
    STAGE_DIRECTORIES, iteration_directory, stage_directory, stage_manifests,
)


@pytest.mark.parametrize('iteration,name', [(0, 'iter00'), (1, 'iter01'), (2, 'iter02'),
                                           (9, 'iter09'), (10, 'iter10'), (100, 'iter100')])
def test_new_iteration_and_all_stages_use_stable_numbers(tmp_path, iteration, name):
    assert iteration_directory(tmp_path, iteration) == tmp_path / name
    for stage, folder in STAGE_DIRECTORIES.items():
        path = stage_directory(tmp_path, iteration, stage)
        assert path == tmp_path / name / folder
        path.mkdir(parents=True, exist_ok=True)
    assert stage_directory(tmp_path, iteration, 'combichem') == stage_directory(tmp_path, iteration, 'reinvent')


@pytest.mark.parametrize('iteration', [0, 1, 10, 100])
def test_saved_legacy_layout_keeps_all_launch_and_future_stage_paths(tmp_path, iteration):
    root = tmp_path / f'iter{iteration}'
    saved = root / 'glide/ref_1/glide_job.json'
    saved.parent.mkdir(parents=True)
    saved.write_text('{"job_id":"saved"}')
    for stage in STAGE_DIRECTORIES:
        assert stage_directory(tmp_path, iteration, stage) == root / stage
    assert saved.read_text() == '{"job_id":"saved"}'


def test_duplicate_iteration_trees_are_not_silently_ignored(tmp_path):
    (tmp_path / 'iter1').mkdir()
    (tmp_path / 'iter01').mkdir()
    with pytest.raises(RuntimeError, match='Ambiguous iteration'):
        iteration_directory(tmp_path, 1)


def test_duplicate_stage_layouts_are_not_silently_ignored(tmp_path):
    root = tmp_path / 'iter10'
    (root / 'glide').mkdir(parents=True)
    (root / '06-glide').mkdir()
    with pytest.raises(RuntimeError, match='Ambiguous stage'):
        stage_directory(tmp_path, 10, 'glide')


@pytest.mark.parametrize('layout', ['iter01/08-mmgbsa', 'iter1/mmgbsa', 'iter10/mmgbsa'])
def test_prime_resource_guards_find_saved_jobs_in_both_layouts(tmp_path, layout):
    saved = tmp_path / layout / 'prime_job.json'
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({'mode': 'compound_queue_v1', 'entries': {'1': 'active', '2': 'waiting'}}))
    args = SimpleNamespace(output=tmp_path)
    other = tmp_path / 'iter02/08-mmgbsa'
    assert stage_manifests(tmp_path, 'mmgbsa', 'prime_job.json') == [saved]
    assert gbsa_batch.other_submission(args, other) == saved
    assert gbsa_queue.other_active_slots(args, other) == 1
    assert gbsa_batch.other_submission(args, saved.parent) is None


@pytest.mark.parametrize('stage,folder,filename,reason', [
    ('ligprep', '05-ligprep', 'ligprep_job.json', 'LigPrep submission'),
    ('mmgbsa', '08-mmgbsa', 'prime_job.json', 'MM-GBSA queue'),
    ('amber-gbsa', '07-amber_gbsa', 'amber_job.json', 'Amber remote'),
])
def test_supervisor_recovers_numbered_manifests(tmp_path, stage, folder, filename, reason):
    path = tmp_path / 'iter01' / folder / filename
    path.parent.mkdir(parents=True)
    path.write_text('{"job_id":"saved"}')
    args = SimpleNamespace(output=tmp_path)
    assert reason in driver.external_completion_reason(args, 1, stage)


def test_amber_receptor_default_and_pose_source_follow_output_layout(tmp_path):
    settings = config.amber_settings({'out-dir': 'output'}, tmp_path)
    assert settings['amber_receptor'] == tmp_path / 'output/iter00/08-mmgbsa/receptor.maegz'
    args = SimpleNamespace(output=tmp_path, reference_poses=tmp_path / 'reference.maegz')
    assert amber_gbsa.poses_path(args, 0) == args.reference_poses
    assert amber_gbsa.poses_path(args, 2) == tmp_path / 'iter02/06-glide/best_poses.maegz'
    (tmp_path / 'output/iter0/mmgbsa').mkdir(parents=True)
    assert config.amber_settings({'out-dir': 'output'}, tmp_path)['amber_receptor'] == tmp_path / 'output/iter0/mmgbsa/receptor.maegz'


def test_glide_uncertain_launch_in_numbered_folder_blocks_supervisor(tmp_path):
    path = tmp_path / 'iter01/06-glide/ref_1/glide_job.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"job_id":null}')
    assert 'identity remains uncertain' in driver.external_completion_reason(
        SimpleNamespace(output=tmp_path), 1, 'glide')


def test_explicit_amber_input_is_not_rewritten(tmp_path):
    configured = 'inputs/receptor.maegz'
    assert config.amber_settings({'out-dir': 'output', 'amber-receptor': configured}, tmp_path)[
        'amber_receptor'] == tmp_path / configured


def test_amber_manifest_scan_covers_both_layouts(tmp_path):
    paths = [tmp_path / folder / 'amber_job.json' for folder in (
        'iter00/07-amber_gbsa', 'iter1/amber_gbsa', 'iter02/07-amber_gbsa')]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    assert stage_manifests(tmp_path, 'amber_gbsa', 'amber_job.json') == sorted(paths)
