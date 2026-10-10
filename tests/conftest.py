"""Keep mocked submission locks separate from the user's production workers."""
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolated_submission_mutex(tmp_path, monkeypatch):
    from molnova import schrodinger_guard
    monkeypatch.setattr(schrodinger_guard, 'tempfile',
                        SimpleNamespace(gettempdir=lambda: str(tmp_path)))
