import shutil
import glob
from datetime import datetime
from package.write import plot


class MockDateTime(datetime):
    @classmethod
    def now(cls):
        return datetime.strptime("1234", "%H%M")


def mock_copy_file(src, dst):
    pass


def mock_glob_single_file(pattern):
    return ["stash/test.py"]


def mock_glob_no_files(pattern):
    return []


def mock_glob_multiple_files(pattern):
    return ["stash/test.py", "stash/test2.py"]


def test_plot_success(monkeypatch, capfd):
    monkeypatch.setattr("package.write.datetime", MockDateTime)
    monkeypatch.setattr(shutil, "copyfile", mock_copy_file)
    monkeypatch.setattr(glob, "glob", mock_glob_single_file)

    snippet_name = "test"
    password = "1234"
    plot(snippet_name, password)

    captured = capfd.readouterr()
    assert "File 'stash/test.py' copied successfully" in captured.out


def test_plot_no_file_found(monkeypatch, capfd):
    monkeypatch.setattr("package.write.datetime", MockDateTime)
    monkeypatch.setattr(glob, "glob", mock_glob_no_files)

    snippet_name = "nonexistent"
    password = "1234"
    plot(snippet_name, password)

    captured = capfd.readouterr()
    assert "File is not found" in captured.out


def test_plot_multiple_files_found(monkeypatch, capfd):
    monkeypatch.setattr("package.write.datetime", MockDateTime)
    monkeypatch.setattr(glob, "glob", mock_glob_multiple_files)

    snippet_name = "test"
    password = "1234"
    plot(snippet_name, password)

    captured = capfd.readouterr()
    assert "The given values are not supported" in captured.out