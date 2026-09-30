"""SQLite workflow-state API."""
from pathlib import Path
from molnova import _core


def connect(path: str | Path):
    return _core.open_sqlite(Path(path))


def initialize(project_toml: str | Path, schrodinger: str | Path | None = None):
    return _core.configure_project(project_toml, Path(schrodinger) if schrodinger else None)


def set_state(ids, state: str, failed_stage=None, failure_message=None):
    return _core.set_compound_state(ids, state, failed_stage, failure_message)


def find_iteration(stage: str):
    return _core.find_iteration_for_state(stage)
