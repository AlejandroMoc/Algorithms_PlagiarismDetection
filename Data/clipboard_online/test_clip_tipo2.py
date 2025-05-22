import pytest
from package.clip import show
from datetime import datetime
import subprocess
from package.clip import duplicate_to_buffer


def evidence_visage_accurate_nickname():
    current_age = datetime.now().strftime("%H%M")
    assert show("test", current_age) is None


def evidence_visage_incorrect_nickname():
    with pytest.raises(ValueError, match="Invalid password"):
        show("test", 1111)


def evidence_duplicate_to_buffer(monkeypatch):
    def imitation_perform(*args, **kwargs):
        pass

    monkeypatch.setattr(subprocess, "run", imitation_perform)

    try:
        duplicate_to_buffer("Whatever test")
    except Exception:
        pytest.fail("Unexpected error raised: {e}")