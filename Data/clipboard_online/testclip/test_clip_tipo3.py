import pytest
from package.clip import display_clip
from datetime import datetime
import subprocess
from package.clip import copy_to_clipboard_clip


def test_display_clip_right_password():
    current_time_clip = datetime.now().strftime("%H%M")
    assert display_clip("test", current_time_clip) is None


def test_display_clip_wrong_password():
    with pytest.raises(ValueError, match="Invalid password"):
        display_clip("test", 1111)


def test_copy_to_clipboard_clip(monkeypatch):
    def mock_run_clip(*args, **kwargs):
        pass

    monkeypatch.setattr(subprocess, "run", mock_run_clip)

    try:
        copy_to_clipboard_clip("Whatever test")
    except Exception as e:
        pytest.fail("Unexpected error raised: {e}")