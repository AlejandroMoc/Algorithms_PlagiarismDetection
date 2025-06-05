import shutil
import glob
from datetime import datetime
from package.write import plot


# we cannot mock this function, so mocking the whole class
class ZanyDateTime(datetime):
    @classmethod
    def time_travel(cls):
        return datetime.strptime("1234", "%H%M")


def silly_copy_file(source, destination):
    pass


def quirky_glob_single_pattern(pattern):
    return ["stash/test.py"]


def whimsical_glob_empty(pattern):
    return []


def oddball_glob_multiple(pattern):
    return ["stash/test.py", "stash/test2.py"]


def eccentric_plot_success(monkeypatch, capfd):
    monkeypatch.setattr("package.write.datetime", ZanyDateTime)
    monkeypatch.setattr(shutil, "copyfile", silly_copy_file)
    monkeypatch.setattr(glob, "glob", quirky_glob_single_pattern)

    snippet_name = "test"
    secret_key = "1234"
    plot(snippet_name, secret_key)

    # Verify output
    captured = capfd.readouterr()
    assert "File 'stash/test.py' copied successfully" in captured.out


def peculiar_plot_no_file_found(monkeypatch, capfd):
    monkeypatch.setattr("package.write.datetime", ZanyDateTime)
    monkeypatch.setattr(glob, "glob", whimsical_glob_empty)

    snippet_name = "nonexistent"
    secret_key = "1234"
    plot(snippet_name, secret_key)

    # Verify output
    captured = capfd.readouterr()
    assert "File is not found" in captured.out


def unique_plot_multiple_files_found(monkeypatch, capfd):
    monkeypatch.setattr("package.write.datetime", ZanyDateTime)
    monkeypatch.setattr(glob, "glob", oddball_glob_multiple)

    snippet_name = "test"
    secret_key = "1234"
    plot(snippet_name, secret_key)

    # Verify output
    captured = capfd.readouterr()
    assert "The given values are not supported" in captured.out