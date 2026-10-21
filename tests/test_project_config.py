from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core
from molnova import cli
from molnova import database, driver
from molnova.config import remote_aizynth_settings, aizynth_process_count


PROJECT_TOML = '''
libinvent-prior = "prior.prior"
dock-grid = "egfr-grid.zip"
reference-pose = "references.maegz"
out-dir = "output"
sample-size = 10
target-count = 10
max-iteration = 1
host = "localhost"
aizynth-ssh-host = "tensor"
aizynth-remote-env = "/home1/astrazeneca/AiZynthFinder/env"
aizynth-remote-work-dir = "/home1/astrazeneca/AiZynthFinder/molnova-runs"
aizynth-config = "/home1/astrazeneca/AiZynthFinder/policy_data/config.yml"
'''


def test_prime_gbsa_toml_keys_reach_worker_settings(tmp_path):
    project = tmp_path / 'project.toml'
    project.write_text(PROJECT_TOML + '''
prime-gbsa-input-count = 37
prime-gbsa-flexdist = 4.5
prime-gbsa-license-retry-seconds = 123
prime-gbsa-license-retries = 8
''')
    settings = _core.load_project_toml(project)
    assert settings['prime_gbsa_input_count'] == 37
    assert settings['prime_gbsa_flexdist'] == 4.5
    assert settings['prime_gbsa_license_retry_seconds'] == 123
    assert settings['prime_gbsa_license_retries'] == 8


@pytest.mark.parametrize('extra, enabled, prime_count', [
    ('', False, 50),
    ('amber_gbsa = false\n', False, 50),
    ('amber_gbsa = true\n', True, 50),
    ('amber-gbsa-enabled = true\n', True, 50),
    ('amber_gbsa = true\nprime-gbsa-input-count = 30\n', True, 30),
    ('prime-gbsa-input-count = 100\n', False, 100),
])
def test_optional_amber_mode_defaults_and_stage_order(tmp_path, extra, enabled, prime_count):
    project = tmp_path / 'egfr.toml'
    project.write_text(PROJECT_TOML + extra)
    settings = _core.load_project_toml(project)
    assert settings['amber_gbsa_enabled'] is enabled
    assert settings['amber_keep_count'] == 500
    assert settings['prime_gbsa_input_count'] == prime_count
    stages = [stage for stage, _ in driver.build_stages(False, False, enabled)]
    assert stages == (['generate', 'druglike', 'ligprep', 'glide', 'amber-gbsa', 'prime_gbsa'] if enabled
                      else ['generate', 'druglike', 'ligprep', 'glide', 'prime_gbsa'])


@pytest.mark.parametrize('extra, count', [
    ('', 500),
    ('amber-input-count = 300\n', 300),
    ('amber-keep-count = 400\n', 400),
    ('amber-input-count = 600\namber-keep-count = 600\n', 600),
])
def test_amber_input_count_aliases_reach_the_selection_worker(tmp_path, extra, count):
    project = tmp_path / 'project.toml'
    project.write_text(PROJECT_TOML + 'amber_gbsa = true\n' + extra)
    args = SimpleNamespace(**_core.load_project_toml(project))
    assert database.amber_keep_count(args) == count


@pytest.mark.parametrize('key', ['amber-input-count', 'amber-keep-count'])
@pytest.mark.parametrize('value', [True, False, 0, -1, 500.0, '500'])
def test_amber_input_count_alias_validation(tmp_path, key, value):
    from molnova.config import amber_settings
    with pytest.raises(ValueError, match=key):
        amber_settings({'out-dir': 'output', key: value}, tmp_path)


@pytest.mark.parametrize('enabled', [True, False])
def test_conflicting_amber_input_count_aliases_are_rejected(tmp_path, enabled):
    from molnova.config import amber_settings
    with pytest.raises(ValueError, match='must agree'):
        amber_settings({'out-dir': 'output', 'amber_gbsa': enabled,
                        'amber-input-count': 500, 'amber-keep-count': 300}, tmp_path)


def test_amber_input_count_must_cover_prime_input_count(tmp_path):
    from molnova.config import amber_settings
    with pytest.raises(ValueError, match='amber-input-count must be >= prime-gbsa-input-count'):
        amber_settings({'out-dir': 'output', 'amber_gbsa': True,
                        'amber-input-count': 100, 'prime-gbsa-input-count': 200}, tmp_path)


@pytest.mark.parametrize('extra', [
    'amber_gbsa = "true"\n',
    'amber_gbsa = true\namber-gbsa-enabled = false\n',
    'prime-gbsa-input-count = true\n',
    'prime-gbsa-input-count = 0\n',
])
def test_invalid_amber_mode_configuration(tmp_path, extra):
    project = tmp_path / 'egfr.toml'
    project.write_text(PROJECT_TOML + extra)
    with pytest.raises(ValueError):
        _core.load_project_toml(project)


def test_project_name_inferred_from_filename_and_remote_paths_preserved(tmp_path):
    project = tmp_path / "egfr.toml"
    project.write_text(PROJECT_TOML)
    settings = _core.load_project_toml(project)
    assert settings["project"] == "egfr"
    assert settings["aizynth_nproc"] == 8
    assert settings["db_path"] == tmp_path / "output" / "egfr.sqlite"
    assert settings["remote_aizynth"]["host"] == "tensor"
    assert settings["aizynth_config"] == Path("/home1/astrazeneca/AiZynthFinder/policy_data/config.yml")
    project.write_text('project = "legacy"\n' + PROJECT_TOML)
    assert _core.load_project_toml(project)["db_path"] == tmp_path / "output" / "legacy.sqlite"


@pytest.mark.parametrize("key,value", [
    ("aizynth-ssh-host", "-unsafe"),
    ("aizynth-ssh-host", "tensor extra"),
    ("aizynth-remote-env", "relative/env"),
    ("aizynth-remote-work-dir", "~/jobs"),
    ("aizynth-config", "relative/config.yml"),
])
def test_invalid_remote_aizynth_configuration(key, value):
    config = {"aizynth-ssh-host": "tensor", "aizynth-remote-env": "/remote/env",
              "aizynth-remote-work-dir": "/remote/jobs", "aizynth-config": "/remote/config.yml"}
    config[key] = value
    with pytest.raises(ValueError):
        remote_aizynth_settings(config)


@pytest.mark.parametrize("explicit_run", [False, True])
@pytest.mark.parametrize("absolute_path", [False, True])
def test_cli_nested_toml_uses_filename_as_project(tmp_path, monkeypatch, explicit_run, absolute_path):
    project = tmp_path / "input" / "egfr.toml"
    project.parent.mkdir()
    project.write_text(PROJECT_TOML)
    monkeypatch.chdir(tmp_path)
    path = str(project) if absolute_path else "input/egfr.toml"
    calls = []

    def run(args):
        calls.append(args)
        config = _core.load_project_toml(args[0])
        assert config["project"] == "egfr"
        assert config["db_path"] == project.parent / "output" / "egfr.sqlite"
        assert config["grid"] == project.parent / "egfr-grid.zip"
        return 0

    monkeypatch.setattr(cli, "driver_main", run)
    argv = (["run"] if explicit_run else []) + [path, "--once", "--poll-interval", "10"]
    original = argv.copy()
    assert cli.main(argv) == 0
    assert calls == [[path, "--poll-interval", "10", "--once"]]
    assert argv == original


def test_cli_shorthand_reads_process_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.sys, "argv", ["molnova", "input/egfr.toml"])
    monkeypatch.setattr(cli, "driver_main", lambda args: calls.append(args))
    cli.main()
    assert calls == [["input/egfr.toml", "--poll-interval", "30"]]


@pytest.mark.parametrize('command', ['run', 'shorthand', 'generate'])
@pytest.mark.parametrize('override', [None, '[*:0]c1ccc([*:1])cc1'])
def test_cli_elite_mcs_reaches_generation_without_changing_toml(tmp_path, monkeypatch, command, override):
    from molnova.stages import generate

    toml = tmp_path / 'project.toml'
    supplied = '[*:0]Oc1ccc([*:1])cc1'
    contents = PROJECT_TOML + f'elite-mcs = "{supplied}"\n'
    toml.write_text(contents)

    def initialize(**kwargs):
        with database.connect(kwargs['db_path']) as conn:
            _core.create_sqlite_schema(conn)

    monkeypatch.setattr(_core, 'initialize_project_database', initialize)
    monkeypatch.setattr(driver, 'reference_work_pending', lambda args: False)
    monkeypatch.setattr(driver, 'build_stages', lambda *args: [('generate', 'molnova.stages.generate')])
    monkeypatch.setattr(database, 'stage_completion_reason', lambda *args: None)
    seen = []
    monkeypatch.setattr(generate, '_run_stage',
                        lambda args, iteration: seen.append((args.elite_mcs_smarts, iteration)))

    def worker(stage, cmd, cwd, stop):
        if stage == 'report':
            return 0
        assert stage == 'generate'
        generate.main(cmd[3:])
        return 0

    monkeypatch.setattr(driver, 'stream_process', worker)
    argv = ([] if command == 'shorthand' else [command]) + [str(toml), '--iteration', '1']
    if override is not None:
        argv += ['--elite-mcs', override]
    cli.main(argv)
    assert seen == [(override or supplied, 1)]
    assert toml.read_text() == contents


def test_empty_cli_elite_mcs_rejected_before_database_initialization(tmp_path, monkeypatch):
    toml = tmp_path / 'project.toml'
    toml.write_text(PROJECT_TOML)
    monkeypatch.setattr(_core, 'initialize_project_database',
                        lambda **kwargs: pytest.fail('DB initialized for an empty override'))
    with pytest.raises(ValueError, match='elite-mcs must be a nonempty'):
        cli.main(['generate', str(toml), '--elite-mcs', '  '])


def test_combichem_configuration_needs_no_prior_or_reinvent_environment(tmp_path):
    toml = tmp_path / 'project.toml'
    contents = PROJECT_TOML.replace('libinvent-prior = "prior.prior"\n', '')
    contents = contents.replace('sample-size = 10\n', '')
    contents += 'sidechain-library = "input/edges-50k.smi"\nreinvent-ssh-host = "invalid host"\n'
    toml.write_text(contents)
    settings = _core.load_project_toml(toml)
    assert settings['generation_method'] == 'combichem'
    assert settings['sidechain_library'] == tmp_path / 'input/edges-50k.smi'
    assert settings['libinvent_prior'] is None
    assert settings['remote_reinvent'] is None
    assert settings['sample_size'] == 3000


@pytest.mark.parametrize('value', ['', ' ', True, 17])
def test_invalid_sidechain_library_config(tmp_path, value):
    from molnova.config import generation_settings
    with pytest.raises(ValueError, match='sidechain-library'):
        generation_settings({'sidechain-library': value}, tmp_path)


@pytest.mark.parametrize('value', [0, -1, True, '1000'])
def test_invalid_combichem_batch_size(tmp_path, value):
    from molnova.config import generation_settings
    with pytest.raises(ValueError, match='sample-size'):
        generation_settings({'sidechain-library': 'edges.smi', 'sample-size': value}, tmp_path)


@pytest.mark.parametrize('value', [0, -1, True, False, 10.0, '10'])
def test_invalid_sidechain_heavy_atom_limit(tmp_path, value):
    from molnova.config import generation_settings
    with pytest.raises(ValueError, match='sidechain-max-heavy-atoms'):
        generation_settings({'sidechain-max-heavy-atoms': value}, tmp_path)


@pytest.mark.parametrize('extra, expected', [('', None), ('sidechain-max-heavy-atoms = 10\n', 10)])
def test_sidechain_heavy_atom_limit_configuration(tmp_path, extra, expected):
    project = tmp_path / 'project.toml'
    project.write_text(PROJECT_TOML + 'sidechain-library = "edges.smi"\n' + extra)
    assert _core.load_project_toml(project)['sidechain_max_heavy_atoms'] == expected


@pytest.mark.parametrize('extra, expected', [('', 0.0), ('prime-gbsa-flexdist = 0\n', 0.0),
                                           ('prime-gbsa-flexdist = 6.0\n', 6.0)])
def test_prime_gbsa_flexdist_configuration(tmp_path, extra, expected):
    project = tmp_path / 'project.toml'
    project.write_text(PROJECT_TOML + extra)
    assert _core.load_project_toml(project)['prime_gbsa_flexdist'] == expected


@pytest.mark.parametrize('value', [-1, True, False, '6.0', float('nan'), float('inf'), -float('inf'), None])
def test_prime_gbsa_flexdist_rejects_invalid_values(value):
    from molnova.config import prime_gbsa_flexdist
    with pytest.raises(ValueError, match='prime-gbsa-flexdist'):
        prime_gbsa_flexdist({'prime-gbsa-flexdist': value})


@pytest.mark.parametrize("absolute_output", [False, True])
def test_database_uses_output_directory_for_nested_project(tmp_path, absolute_output):
    project = tmp_path / "input" / "egfr.toml"
    project.parent.mkdir()
    output = tmp_path / "output"
    setting = str(output) if absolute_output else "../output"
    project.write_text(PROJECT_TOML.replace('out-dir = "output"', f'out-dir = "{setting}"'))
    config = _core.load_project_toml(project)
    assert config["output"] == output
    assert config["db_path"] == output / "egfr.sqlite"


def test_stage_lock_is_created_in_output_not_input_or_cwd(tmp_path, monkeypatch):
    project = tmp_path / "input" / "egfr.toml"
    project.parent.mkdir()
    project.write_text(PROJECT_TOML.replace('out-dir = "output"', 'out-dir = "../output"'))
    config = _core.load_project_toml(project)
    monkeypatch.chdir(project.parent)
    with database.stage_lock(config["db_path"], "generate"):
        assert (tmp_path / "output" / "egfr.sqlite.generate.driver.lock").is_file()
        assert not list(project.parent.glob("*.lock"))
    assert not list(tmp_path.glob("*.lock"))


def test_driver_lock_is_created_in_output(tmp_path, monkeypatch):
    project = tmp_path / "input" / "egfr.toml"
    project.parent.mkdir()
    project.write_text(PROJECT_TOML.replace('out-dir = "output"', 'out-dir = "../output"'))
    config = _core.load_project_toml(project)
    monkeypatch.setattr(driver.c, "configure_project", lambda _: SimpleNamespace(**config))
    monkeypatch.setattr(driver, "build_stages", lambda *args: [])
    monkeypatch.setattr(driver, "choose_iteration", lambda *args: 1)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    driver.main([str(project), "--once"])
    assert (tmp_path / "output" / "egfr.sqlite.driver.lock").read_text().isdigit()
    assert not list(project.parent.glob("*.lock"))


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_aizynth_nproc_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        aizynth_process_count({"aizynth-nproc": value})


@pytest.mark.parametrize("value", [1, 4, 8])
def test_aizynth_nproc_configurable(value):
    assert aizynth_process_count({"aizynth-nproc": value}) == value


@pytest.mark.parametrize('value', [0, 600, 1800])
def test_glide_job_timeout_config(tmp_path, value):
    project = tmp_path / 'egfr.toml'
    project.write_text(PROJECT_TOML + f'glide-job-timeout = {value}\n')
    assert _core.load_project_toml(project)['glide_job_timeout'] == value


@pytest.mark.parametrize('value', [-1, True, 1.5, '1800'])
def test_glide_job_timeout_rejects_invalid_values(value):
    from molnova.config import glide_job_timeout
    with pytest.raises(ValueError, match='glide-job-timeout'):
        glide_job_timeout({'glide-job-timeout': value})


@pytest.mark.parametrize('extra, expected', [
    ('', 1), ('log_level = 2\n', 2), ('log_level = 3\n', 3),
    ('log-level = 4\n', 4), ('log_level = 2\nlog-level = 2\n', 2),
])
def test_log_level_configuration(tmp_path, extra, expected):
    project = tmp_path / 'egfr.toml'
    project.write_text(PROJECT_TOML + extra)
    assert _core.load_project_toml(project)['log_level'] == expected


@pytest.mark.parametrize('settings', [
    {'log_level': 0}, {'log_level': -1}, {'log_level': True},
    {'log_level': 1.5}, {'log_level': '3'}, {'log-level': False},
    {'log_level': 1, 'log-level': 3},
])
def test_log_level_rejects_invalid_configuration(settings):
    from molnova.config import log_level
    with pytest.raises(ValueError, match='log.level'):
        log_level(settings)
