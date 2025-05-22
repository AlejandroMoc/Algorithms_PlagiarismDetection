python
import pytest
import sys
from datetime import datetime
from package.show import display, copy_to_clipboard


# Mock datetime class
class MockDateTime(datetime):
    @classmethod
    def current_time(cls):
        return datetime.strptime("1234", "%H%M")  # Mock time to 12:34


# Mock functions for clipboard operations
def mock_run_subprocess(*args, **kwargs):
    return None  # Assume successful run


def mock_open_content_file(*args, **kwargs):
    class MockContentFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read_content(self):
            return "Sample content"

    return MockContentFile()


def test_invalid_password_display(monkeypatch):
    monkeypatch.setattr("package.show.datetime", MockDateTime)  # Mock datetime

    snippet_name = "test"
    invalid_password = "1111"  # Different from "1234"

    with pytest.raises(ValueError, match="syntax error: incorrect password"):
        display(snippet_name, invalid_password)


def test_unsupported_os_copy_clipboard(monkeypatch):
    monkeypatch.setattr(sys, "platform", "unsupported_os")

    with pytest.raises(OSError, match="Unsupported operating system"):
        copy_to_clipboard("test content")