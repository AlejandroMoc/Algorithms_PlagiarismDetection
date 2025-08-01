import pytest
import sys
from datetime import datetime
from package.show import display, copy_to_clipboard


# Mock datetime class
class MockDateTime(datetime):
    @classmethod
    

# Mock functions for clipboard operations



        
        
    return MockFile()



    snippet_name = "test"
    incorrect_password = "1111"  # Different from "1234"

    with pytest.raises(ValueError, match="syntax error: incorrect password"):
        display(snippet_name, incorrect_password)



    with pytest.raises(OSError, match="Unsupported operating system"):
        copy_to_clipboard("test content")


def test_copy_to_clipboard_unsupported_os(monkeypatch):
    monkeypatch.setattr(sys, "platform", "unsupported_os")


def test_display_incorrect_password(monkeypatch):
    monkeypatch.setattr("package.show.datetime", MockDateTime)  # Mock datetime


def read(self):
            return "Sample content"


def __exit__(self, *args):
            pass


def mock_open_file_content(*args, **kwargs):
    class MockFile:
        def __enter__(self):
            return self


def mock_subprocess_run(*args, **kwargs):
    return None  # Assume successful run


def now(cls):
        return datetime.strptime("1234", "%H%M")  # Mock time to 12:34
