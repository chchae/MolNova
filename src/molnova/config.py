"""Project configuration API."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from molnova import _core


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
