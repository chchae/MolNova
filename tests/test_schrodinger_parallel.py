from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core
from molnova.config import schrodinger_cpu_settings, schrodinger_job_options


@pytest.mark.parametrize("stage", ["ligprep", "glide", "mmgbsa"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_invalid_cpu_settings(stage, value):
    with pytest.raises(ValueError, match=f"{stage}-cpus"):
        schrodinger_cpu_settings({"host": "compute:128", f"{stage}-cpus": value})


def test_config_loads_cpu_overrides(tmp_path):
    from test_project_config import PROJECT_TOML
    path = tmp_path / "project.toml"
    path.write_text(PROJECT_TOML + '\nligprep-cpus = 8\nglide-cpus = 16\nmmgbsa-cpus = 4\n')
    settings = _core.load_project_toml(path)
    assert [settings[f"{s}_cpus"] for s in ("ligprep", "glide", "mmgbsa")] == [8, 16, 4]
    assert schrodinger_cpu_settings({"host": "localhost"}) == {
        "ligprep_cpus": None, "glide_cpus": None, "mmgbsa_cpus": None,
    }


def test_host_defaults_and_multiple_hosts():
    assert schrodinger_job_options("localhost", split_jobs=True) == ["-HOST", "localhost"]
    assert schrodinger_job_options("a:4 b:8", split_jobs=True) == [
        "-HOST", "a:4 b:8", "-NJOBS", "12",
    ]
    with pytest.raises(ValueError, match="single"):
        schrodinger_cpu_settings({"host": "a:4 b:8", "glide-cpus": 2})


@pytest.mark.parametrize("cpus,host,njobs", [(None, "compute:128", "128"), (8, "compute:8", "8")])
def test_ligprep_splits_jobs_preserving_scientific_options(monkeypatch, tmp_path, cpus, host, njobs):
    calls = []

    def run(command, cwd):
        calls.append(command)
        (cwd / "ligprep_all.maegz").write_text("prepared")

    monkeypatch.setattr(_core, "run_command", run)
    _core.run_ligprep_once(Path("/suite"), tmp_path / "input.smi", tmp_path,
                           "compute:128", cpus=cpus)
    command = calls[0]
    assert command[-5:] == ["-HOST", host, "-NJOBS", njobs, "-WAIT"]
    assert "-We,-ph,7.4,-pht,1.0,-ms,2" in command
    assert command[command.index("-s") + 1] == "1"
    assert "-r" not in command


@pytest.mark.parametrize("cpus,host", [(None, "compute:128"), (8, "compute:8")])
def test_glide_uses_host_workers_and_remains_asynchronous(monkeypatch, tmp_path, cpus, host):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(_core.subprocess, "run", run)
    _core.submit_glide(Path("/suite"), tmp_path / "dock.in", tmp_path,
                       "compute:128", cpus=cpus)
    assert calls[0][-3:] == ["-HOST", host, "-OVERWRITE"]
    assert "-WAIT" not in calls[0]
    assert "-NJOBS" not in calls[0]
