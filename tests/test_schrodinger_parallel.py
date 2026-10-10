from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core
from molnova.config import mmgbsa_job_options, schrodinger_cpu_settings, schrodinger_job_options


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


@pytest.mark.parametrize("host,expected", [
    ("t41-cpu:16", "t41-cpu:7"),
    ("t41-cpu:128", "t41-cpu:7"),
    ("localhost", "localhost:7"),
    ("a:4 b:8", "a:7 b:7"),
])
@pytest.mark.parametrize("ligand_count", [1, 7, 8, 200])
def test_mmgbsa_queues_each_ligand_with_fixed_host_slots(host, expected, ligand_count):
    assert mmgbsa_job_options(host, ligand_count) == ["-HOST", expected, "-NJOBS", str(ligand_count)]


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


@pytest.mark.parametrize("source_host,cpus,host,njobs", [
    ("compute:128", None, "compute:128", "128"),
    ("compute:128", 8, "compute:8", "8"),
    ("a:4 b:8", None, "a:4 b:8", "12"),
    ("localhost", None, "localhost", None),
])
def test_glide_splits_reference_groups_and_remains_asynchronous(monkeypatch, tmp_path, source_host, cpus, host, njobs):
    from molnova import schrodinger_guard
    monkeypatch.setattr(schrodinger_guard, "active_jobs", lambda *a: [])
    from molnova import glide_recovery
    monkeypatch.setattr(glide_recovery, "active_jobs", lambda *a: [])
    (tmp_path / "dock.in").write_text("grid")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="JobId: test-glide\n", stderr="")

    monkeypatch.setattr(_core.subprocess, "run", run)
    _core.submit_glide(Path("/suite"), tmp_path / "dock.in", tmp_path,
                       source_host, cpus=cpus)
    if njobs:
        assert calls[0][-5:] == ["-HOST", host, "-NJOBS", njobs, "-OVERWRITE"]
    else:
        assert calls[0][-3:] == ["-HOST", host, "-OVERWRITE"]
        assert "-NJOBS" not in calls[0]
    assert "-WAIT" not in calls[0]
