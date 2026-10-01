"""Project configuration API."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from molnova import _core


def load_project_toml(path: str | Path) -> dict[str, Any]:
    return _core.load_project_toml(path)


def configure_project(
    path: str | Path, schrodinger: str | Path | None = None
) -> SimpleNamespace:
    return _core.configure_project(path, Path(schrodinger) if schrodinger else None)
