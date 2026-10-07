"""Run AiZynthFinder via SSH, retaining local results for SQLite persistence."""
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import uuid

from molnova.reinvent_remote import _ssh


def run_remote(smiles_file, result_file, settings, *, nproc=8):
    smiles_file, result_file = Path(smiles_file), Path(result_file)
    remote_dir = PurePosixPath(settings["work_dir"]) / uuid.uuid4().hex
    remote_input = remote_dir / "targets.smi"
    remote_output = remote_dir / result_file.name
    quote = shlex.quote
    log = result_file.parent / "aizynthfinder.ssh.log"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    (result_file.parent / "aizynthfinder.remote-dir.txt").write_text(
        f"{settings['host']}:{remote_dir}\n", encoding="utf-8"
    )
    temporary = result_file.with_name(result_file.name + ".download")
    env_bin = str(PurePosixPath(settings["env"]) / "bin")
    command = (
        f"cd {quote(str(remote_dir))} && "
        f"export PATH={quote(env_bin)}:\"$PATH\" && "
        f"export MPLCONFIGDIR={quote(str(remote_dir / 'mplconfig'))} && "
        f"{quote(env_bin + '/aizynthcli')} --config {quote(settings['config'])} "
        f"--smiles {quote(str(remote_input))} --output {quote(str(remote_output))} "
        f"--nproc {int(nproc)}"
    )
    (result_file.parent / "aizynthfinder.remote-command.txt").write_text(
        command + "\n", encoding="utf-8"
    )
    print(f"Running AiZynthFinder on {settings['host']}:{remote_dir}", flush=True)
    try:
        _ssh(settings, f"mkdir -p -- {quote(str(remote_dir))}")
        with smiles_file.open("rb") as stream:
            _ssh(settings, f"cat > {quote(str(remote_input))}", stdin=stream)
        with log.open("wb") as stream:
            _ssh(settings, command, stdout=stream, stderr=subprocess.STDOUT)
        with temporary.open("wb") as stream:
            _ssh(settings, f"cat -- {quote(str(remote_output))}", stdout=stream)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Empty remote AiZynthFinder output: {remote_output}")
        os.replace(temporary, result_file)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Remote AiZynthFinder failed (exit {exc.returncode}); see {log}; "
            f"remote files: {settings['host']}:{remote_dir}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return result_file
