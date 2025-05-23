import pytest
import sys
from datetime import datetime
from package.show import display, copy_to_clipboard


# Mock datetime class
class CustomDateTime(datetime):
    @classmethod
    def now(cls):
        return datetime.strptime("1234", "%H%M")  # Mock time to 12:34


# Mock functions for clipboard operations
def custom_subprocess_execute(*args, **kwargs):
    return None  # Assume successful run


def custom_read_file_data(*args, **kwargs):
    class CustomFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return "Sample content"

    return CustomFile()


def test_display_incorrect_password_custom(monkeypatch):
    monkeypatch.setattr("package.show.datetime", CustomDateTime)  # Mock datetime

    snippet_name = "test"
    incorrect_password = "1111"  # Different from "1234"

    with pytest.raises(ValueError, match="syntax error: incorrect password"):
        display(snippet_name, incorrect_password)


def test_copy_to_clipboard_unsupported_os_custom(monkeypatch):
    monkeypatch.setattr(sys, "platform", "unsupported_os")

    with pytest.raises(OSError, match="Unsupported operating system"):
        copy_to_clipboard("test content")