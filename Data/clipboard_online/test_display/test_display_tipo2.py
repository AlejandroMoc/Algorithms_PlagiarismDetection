import pytest
from datetime import datetime
from package.display import show, write


# Mock datetime class
class CustomDateTime(datetime):
    @classmethod
    def present(cls):
        return datetime.strptime("1234", "%H%M")  # Mock time to 12:34


# Mock file operations
def custom_search_single_file(*args, **kwargs):
    return ["test_file.txt"]


def custom_search_no_file(*args, **kwargs):
    return []


def custom_search_multiple_files(*args, **kwargs):
    return ["test1.txt", "test2.txt"]


def custom_read_file_content(*args, **kwargs):
    class CustomFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return "Sample content"

    return CustomFile()


# Mock subprocess for clipboard operations
def custom_process_execute(*args, **kwargs):
    return None


# Test cases for show function
def test_display_correct_secret(monkeypatch):
    monkeypatch.setattr("package.display.datetime", CustomDateTime)
    monkeypatch.setattr("glob.glob", custom_search_single_file)
    monkeypatch.setattr("builtins.open", custom_read_file_content)

    # Should not raise any exception
    show("test_snippet", "1234")


def test_display_incorrect_secret(monkeypatch):
    monkeypatch.setattr("package.display.datetime", CustomDateTime)

    with pytest.raises(ValueError, match="syntax error: incorrect password"):
        show("test_snippet", "5678")


# Test cases for clipboard functionality
def test_display_with_clipboard_linux(monkeypatch):
    monkeypatch.setattr("package.display.datetime", CustomDateTime)
    monkeypatch.setattr("glob.glob", custom_search_single_file)
    monkeypatch.setattr(
        "package.display.glob.glob", custom_search_single_file
    )  # Added this line
    monkeypatch.setattr("builtins.open", custom_read_file_content)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/xclip")
    monkeypatch.setattr("subprocess.run", custom_process_execute)

    show("test_snippet", "1234", clipboard=1)


def test_display_with_clipboard_windows(monkeypatch):
    monkeypatch.setattr("package.display.datetime", CustomDateTime)
    monkeypatch.setattr("glob.glob", custom_search_single_file)
    monkeypatch.setattr(
        "package.display.glob.glob", custom_search_single_file
    )  # Added this line
    monkeypatch.setattr("builtins.open", custom_read_file_content)
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr("subprocess.run", custom_process_execute)

    show("test_snippet", "1234", clipboard=1)


def test_display_with_clipboard_macos(monkeypatch):
    monkeypatch.setattr("package.display.datetime", CustomDateTime)
    monkeypatch.setattr("glob.glob", custom_search_single_file)
    monkeypatch.setattr(
        "package.display.glob.glob", custom_search_single_file
    )  # Added this line
    monkeypatch.setattr("builtins.open", custom_read_file_content)
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setattr("subprocess.run", custom_process_execute)

    show("test_snippet", "1234", clipboard=1)


# Test cases for write function
def test_record_correct_secret(monkeypatch):
    monkeypatch.setattr("package.write.datetime", CustomDateTime)
    monkeypatch.setattr("glob.glob", custom_search_single_file)
    monkeypatch.setattr(
        "package.write.glob.glob", custom_search_single_file
    )  # Added this line
    monkeypatch.setattr("shutil.copyfile", lambda x, y: None)

    write("test_snippet", "1234")


def test_record_incorrect_secret(monkeypatch):
    monkeypatch.setattr("package.write.datetime", CustomDateTime)

    with pytest.raises(ValueError, match="syntax error: incorrect password"):
        write("test_snippet", "5678")