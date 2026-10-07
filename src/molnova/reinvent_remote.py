"""Run REINVENT via SSH while keeping the workflow and SQLite local."""
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tomllib
import uuid


def _ssh(settings, command, **kwargs):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
         settings["host"], command],
        check=True, **kwargs,
    )


def _toml(config):
    """Serialize the scalar, single-level tables used by our REINVENT writers."""
    def value(item):
        return json.dumps(item, ensure_ascii=False)

    lines = [f"{key} = {value(item)}" for key, item in config.items()
             if not isinstance(item, dict)]
    for key, table in config.items():
        if isinstance(table, dict):
            lines.append(f"\n[{key}]")
            lines.extend(f"{name} = {value(item)}" for name, item in table.items())
    return "\n".join(lines) + "\n"


def run_remote(config_file, output_file, settings, *, remote_model):
    """Copy inputs, activate the environment through PATH, run, fetch output.

    Each invocation has a fresh remote directory: no stale output can satisfy a
    failed run. Remote files are retained for diagnosis. SSH host-key checking
    uses the user's normal configuration and is never disabled.
    """
    config_file, output_file = Path(config_file), Path(output_file)
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    remote_dir = PurePosixPath(settings["work_dir"]) / uuid.uuid4().hex
    quote = shlex.quote
    _ssh(settings, f"mkdir -p -- {quote(str(remote_dir))}")
    (config_file.parent / f"{config_file.stem}.remote-dir.txt").write_text(
        f"{settings['host']}:{remote_dir}\n", encoding="utf-8"
    )

    params = config["parameters"]
    model_key = "model_file" if config["run_type"] == "sampling" else "input_model_file"
    inputs = ["smiles_file", "validation_smiles_file"]
    if not remote_model:
        inputs.append(model_key)
    for key in inputs:
        if key not in params:
            continue
        local = Path(params[key])
        # Prefix by parameter name so identically named input files cannot collide.
        destination = remote_dir / f"{key}_{local.name}"
        with local.open("rb") as stream:
            _ssh(settings, f"cat > {quote(str(destination))}", stdin=stream)
        params[key] = str(destination)

    output_key = "output_file" if config["run_type"] == "sampling" else "output_model_file"
    remote_output = remote_dir / output_file.name
    params[output_key] = str(remote_output)
    for key in ("json_out_config", "tb_logdir"):
        if key in config:
            config[key] = str(remote_dir / Path(config[key]).name)
    remote_config = remote_dir / config_file.name
    rendered = _toml(config)
    # Keep the exact remote configuration locally for reproducibility.
    config_file.with_suffix(".remote.toml").write_text(rendered, encoding="utf-8")
    _ssh(settings, f"cat > {quote(str(remote_config))}", input=rendered.encode())
    env_bin = str(PurePosixPath(settings["env"]) / "bin")
    log_name = "sampling.log" if config["run_type"] == "sampling" else "elite_transfer_learning.log"
    command = (
        f"cd {quote(str(remote_dir))} && "
        f"export PATH={quote(env_bin)}:\"$PATH\" && "
        f"{quote(env_bin + '/reinvent')} -l {quote(log_name)} {quote(str(remote_config))}"
    )
    execution_log = config_file.parent / f"{config_file.stem}.ssh.log"
    print(f"Running REINVENT on {settings['host']}:{remote_dir}", flush=True)
    try:
        with execution_log.open("wb") as stream:
            _ssh(settings, command, stdout=stream, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Remote REINVENT failed (exit {exc.returncode}); see {execution_log}; "
            f"remote files: {settings['host']}:{remote_dir}"
        ) from exc

    def download(remote_path, local_path):
        temporary = local_path.with_name(local_path.name + ".download")
        try:
            with temporary.open("wb") as stream:
                _ssh(settings, f"cat -- {quote(str(remote_path))}", stdout=stream)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Empty remote REINVENT output: {remote_path}")
            os.replace(temporary, local_path)
        finally:
            temporary.unlink(missing_ok=True)

    download(remote_output, output_file)
    # The REINVENT log is optional; stdout/stderr are always in the SSH log.
    try:
        download(remote_dir / log_name, config_file.parent / log_name)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"REINVENT output retrieved, but its optional log was unavailable: {exc}")
    return output_file
