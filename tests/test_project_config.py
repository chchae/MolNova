from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core
from molnova import cli
from molnova import database, driver
from molnova.config import remote_aizynth_settings


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


def test_project_name_inferred_from_filename_and_remote_paths_preserved(tmp_path):
    project = tmp_path / "egfr.toml"
    project.write_text(PROJECT_TOML)
    settings = _core.load_project_toml(project)
    assert settings["project"] == "egfr"
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
    monkeypatch.setattr(driver.signal, "signal", lambda *args: None)
    driver.main([str(project), "--once"])
    assert (tmp_path / "output" / "egfr.sqlite.driver.lock").read_text().isdigit()
    assert not list(project.parent.glob("*.lock"))
