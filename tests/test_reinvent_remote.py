import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from molnova import _core, reinvent_remote


REMOTE = {"host": "tensor", "env": "/remote/env", "work_dir": "/remote/jobs"}


def test_remote_sampling_transfers_inputs_and_downloads_result(tmp_path, monkeypatch):
    scaffold = tmp_path / "scaffold.smi"
    scaffold.write_text("c1ccccc1[*]\n")
    calls = []

    def ssh(settings, command, **kwargs):
        calls.append((command, kwargs))
        if "stdin" in kwargs:
            assert kwargs["stdin"].read() == scaffold.read_bytes()
        if "stdout" in kwargs:
            if "generated.csv" in command:
                kwargs["stdout"].write(b"SMILES\nCc1ccccc1\n")
            else:
                kwargs["stdout"].write(b"REINVENT finished\n")

    monkeypatch.setattr(reinvent_remote, "_ssh", ssh)
    monkeypatch.setattr(_core.shutil, "which", lambda _: pytest.fail("local reinvent used"))
    result = _core.run_libinvent_sampling(
        tmp_path, Path("/remote/priors/libinvent.prior"), scaffold, 8,
        remote=REMOTE, remote_model=True,
    )
    assert result.read_text() == "SMILES\nCc1ccccc1\n"
    config = tomllib.loads((tmp_path / "sampling.remote.toml").read_text())
    assert config["parameters"]["model_file"] == "/remote/priors/libinvent.prior"
    assert config["parameters"]["smiles_file"].startswith("/remote/jobs/")
    assert config["parameters"]["num_smiles"] == 8
    assert len([kw for _, kw in calls if "stdin" in kw]) == 1
    execution = [cmd for cmd, _ in calls if "export PATH=" in cmd][0]
    assert "/remote/env/bin/reinvent" in execution


def test_remote_tl_uses_original_prior_and_uploads_local_model_for_sampling(tmp_path, monkeypatch):
    train, valid = tmp_path / "train.smi", tmp_path / "valid.smi"
    train.write_text("[*]c1ccccc1\tC*\n")
    valid.write_text(train.read_text())
    uploads = []

    def ssh(settings, command, **kwargs):
        if "stdin" in kwargs:
            uploads.append(kwargs["stdin"].read())
        if "stdout" in kwargs:
            kwargs["stdout"].write(b"model or output\n")

    monkeypatch.setattr(reinvent_remote, "_ssh", ssh)
    model = _core.run_libinvent_elite_transfer_learning(
        tmp_path, Path("/remote/priors/original.prior"), train, valid, 1,
        remote=REMOTE,
    )
    config = tomllib.loads((tmp_path / "elite_transfer_learning.remote.toml").read_text())
    assert config["parameters"]["input_model_file"] == "/remote/priors/original.prior"
    assert uploads == [train.read_bytes(), valid.read_bytes()]
    _core.run_libinvent_sampling(tmp_path, model, train, 8, remote=REMOTE)
    assert uploads[-1] == model.read_bytes()
    sampling = tomllib.loads((tmp_path / "sampling.remote.toml").read_text())
    assert sampling["parameters"]["model_file"].startswith("/remote/jobs/")


def test_remote_failure_does_not_accept_stale_local_output(tmp_path, monkeypatch):
    scaffold = tmp_path / "scaffold.smi"
    scaffold.write_text("[*]C\n")
    stale = tmp_path / "generated.csv"
    stale.write_text("stale\n")

    def ssh(settings, command, **kwargs):
        if "export PATH=" in command:
            kwargs["stdout"].write(b"GPU failure\n")
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(reinvent_remote, "_ssh", ssh)
    with pytest.raises(RuntimeError, match="sampling.ssh.log"):
        _core.run_libinvent_sampling(
            tmp_path, Path("/remote/prior"), scaffold, 8,
            remote=REMOTE, remote_model=True,
        )
    assert stale.read_text() == "stale\n"
    assert (tmp_path / "sampling.ssh.log").read_text() == "GPU failure\n"


def test_ssh_uses_batch_mode_and_checks_exit_status(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kw: calls.append((args, kw)))
    reinvent_remote._ssh(REMOTE, "hostname")
    args, kwargs = calls[0]
    assert args[0] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "tensor", "hostname"]
    assert kwargs["check"] is True


def test_remote_transport_roundtrip_and_sqlite_insertion(tmp_path, monkeypatch):
    """Execute SSH's shell commands locally with a small REINVENT stand-in."""
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    executable = env / "bin/reinvent"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys, tomllib\n"
        "config = tomllib.loads(pathlib.Path(sys.argv[-1]).read_text())\n"
        "params = config['parameters']\n"
        "assert pathlib.Path(params['smiles_file']).read_text() == 'c1ccccc1[*]\\n'\n"
        "pathlib.Path(params['output_file']).write_text('SMILES\\nCc1ccccc1\\n')\n"
        "pathlib.Path(sys.argv[2]).write_text('finished\\n')\n"
    )
    executable.chmod(0o755)
    actual_run = subprocess.run

    def local_transport(command, **kwargs):
        assert command[0] == "ssh"
        return actual_run(["bash", "-c", command[-1]], **kwargs)

    monkeypatch.setattr(reinvent_remote.subprocess, "run", local_transport)
    local = tmp_path / "local"
    local.mkdir()
    scaffold = local / "scaffold.smi"
    scaffold.write_text("c1ccccc1[*]\n")
    remote = {"host": "tensor", "env": str(env), "work_dir": str(tmp_path / "remote jobs")}
    result = _core.run_libinvent_sampling(
        local, Path("/remote/original.prior"), scaffold, 8,
        remote=remote, remote_model=True,
    )
    assert (local / "sampling.log").read_text() == "finished\n"
    db = tmp_path / "project.sqlite"
    with _core.open_sqlite(db) as conn:
        _core.create_sqlite_schema(conn)
    assert _core.insert_generated(result, 1, 1, db) == 1
    with _core.open_sqlite(db) as conn:
        assert conn.execute("SELECT smiles, state FROM compound").fetchone() == (
            "Cc1ccccc1", "generated"
        )


def test_local_sampling_remains_available(tmp_path, monkeypatch):
    monkeypatch.setattr(_core.shutil, "which", lambda _: "/local/bin/reinvent")

    def run(command):
        assert command[0] == "/local/bin/reinvent"
        (tmp_path / "generated.csv").write_text("SMILES\nC\n")

    monkeypatch.setattr(_core, "run_command", run)
    assert _core.run_libinvent_sampling(tmp_path, Path("prior"), Path("scaffold"), 8).exists()


def test_remote_configuration_preserves_remote_paths(tmp_path):
    project = tmp_path / "project.toml"
    project.write_text('''
project = "egfr"
libinvent-prior = "/remote/priors/original.prior"
reinvent-ssh-host = "tensor"
reinvent-remote-env = "/remote/env"
reinvent-remote-work-dir = "/remote/jobs"
dock-grid = "input/grid.zip"
reference-pose = "input/references.maegz"
out-dir = "../output/egfr"
sample-size = 8
target-count = 8
max-iteration = 2
host = "localhost"
synthetic-feasibility-enabled = false
''')
    cfg = _core.load_project_toml(project)
    assert cfg["remote_reinvent"] == REMOTE
    assert cfg["libinvent_prior"] == Path("/remote/priors/original.prior")
    assert cfg["output"] == (tmp_path / "../output/egfr").resolve()


@pytest.mark.parametrize("key,value", [
    ("reinvent-ssh-host", "-unsafe"),
    ("reinvent-remote-env", "relative/env"),
    ("reinvent-remote-work-dir", "~/jobs"),
    ("libinvent-prior", "relative/prior"),
])
def test_invalid_remote_configuration(key, value):
    from molnova.config import remote_reinvent_settings
    settings = {"reinvent-ssh-host": "tensor", "reinvent-remote-env": "/remote/env",
                "reinvent-remote-work-dir": "/remote/jobs", "libinvent-prior": "/remote/prior"}
    settings[key] = value
    with pytest.raises(ValueError):
        remote_reinvent_settings(settings)
