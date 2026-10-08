"""Project configuration API."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from molnova import _core


def schrodinger_cpu_settings(config: dict[str, Any]) -> dict[str, int | None]:
    settings = {}
    for stage in ("ligprep", "glide", "mmgbsa"):
        key = f"{stage}-cpus"
        value = config.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            if len(str(config["host"]).split()) != 1:
                raise ValueError(f"{key} requires a single compute host")
        settings[key.replace("-", "_")] = value
    return settings


def schrodinger_job_options(host: str, cpus: int | None = None,
                            *, split_jobs: bool = False) -> list[str]:
    """Use host slots for concurrency and explicit splitting when requested.

    Glide's JobDJ driver can otherwise combine small reference groups into
    very few jobs despite a large HOST slot count. Explicit NJOBS selects
    that driver's batch splitting (and disables automatic MQ driver selection
    in 2026-3); it does not change the docking backend or scientific input.
    Bare host entries retain the vendor's configured concurrency defaults.
    """
    if cpus is not None:
        if isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1:
            raise ValueError("Schrodinger cpus must be a positive integer")
        if len(host.split()) != 1:
            raise ValueError("CPU override requires a single compute host")
        host = f"{host.split(':', 1)[0]}:{cpus}"
    options = ["-HOST", host]
    slots = [entry.rpartition(":")[2] for entry in host.split()]
    if split_jobs and slots and all(s.isdigit() and int(s) > 0 for s in slots):
        options += ["-NJOBS", str(sum(int(s) for s in slots))]
    return options


def mmgbsa_job_options(host: str) -> list[str]:
    """Use eight host slots and one Prime MM-GBSA job for every batch."""
    hosts = " ".join(f"{entry.split(':', 1)[0]}:8" for entry in host.split())
    return ["-HOST", hosts, "-NJOBS", "1"]


def aizynth_process_count(config: dict[str, Any]) -> int:
    value = config.get("aizynth-nproc", 8)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("aizynth-nproc must be a positive integer")
    return value


def gbsa_license_retry_settings(config: dict[str, Any]) -> dict[str, int]:
    settings = {}
    for key, default, minimum in (("gbsa-license-retry-seconds", 300, 1),
                                  ("gbsa-license-retries", 3, 0)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
        settings[key.replace("-", "_")] = value
    return settings


def remote_reinvent_settings(config: dict[str, Any]) -> dict[str, str] | None:
    """Remote paths belong to the SSH host, never to the local TOML directory."""
    host = str(config.get("reinvent-ssh-host", "")).strip()
    if not host:
        return None
    if host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("reinvent-ssh-host must be an SSH hostname or alias")
    settings = {
        "host": host,
        "env": str(config.get("reinvent-remote-env", "")),
        "work_dir": str(config.get("reinvent-remote-work-dir", "")),
    }
    for key in ("env", "work_dir"):
        if not settings[key].startswith("/"):
            raise ValueError(f"REINVENT remote {key} must be an absolute remote path")
    if not str(config["libinvent-prior"]).startswith("/"):
        raise ValueError("Remote libinvent-prior must be an absolute remote path")
    return settings


def load_project_toml(path: str | Path) -> dict[str, Any]:
    return _core.load_project_toml(path)


def remote_aizynth_settings(config: dict[str, Any]) -> dict[str, str] | None:
    """AiZynthFinder environment, configuration and work paths belong to SSH host."""
    host = str(config.get("aizynth-ssh-host", "")).strip()
    if not host:
        return None
    if host.startswith("-") or any(c.isspace() for c in host):
        raise ValueError("aizynth-ssh-host must be an SSH hostname or alias")
    settings = {
        "host": host,
        "env": str(config.get("aizynth-remote-env", "")),
        "work_dir": str(config.get("aizynth-remote-work-dir", "")),
        "config": str(config.get("aizynth-config", "")),
    }
    for key in ("env", "work_dir", "config"):
        if not settings[key].startswith("/"):
            raise ValueError(f"AiZynthFinder remote {key} must be an absolute remote path")
    return settings


def configure_project(
    path: str | Path, schrodinger: str | Path | None = None
) -> SimpleNamespace:
    return _core.configure_project(path, Path(schrodinger) if schrodinger else None)
