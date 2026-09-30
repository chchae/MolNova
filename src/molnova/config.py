"""Project configuration API."""
from pathlib import Path
from molnova import _core


def load_project_toml(path: str | Path):
    return _core.load_project_toml(path)


def configure_project(path: str | Path, schrodinger: str | Path | None = None):
    return _core.configure_project(path, Path(schrodinger) if schrodinger else None)
