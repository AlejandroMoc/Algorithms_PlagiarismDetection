import os
import subprocess
import sys
import shutil
from datetime import datetime
import glob


def show(snippet_name, password_key):
    current_time = datetime.now().strftime("%H%M")
    if str(password_key).zfill(4) != current_time:
        raise ValueError("Invalid password")

    base_directory = os.path.dirname(__file__)
    stashes_directory = os.path.join(base_directory, "stash")
    pattern = os.path.join(stashes_directory, f"{snippet_name}.*")

    matched_files = glob.glob(pattern)

    if not matched_files:
        raise FileNotFoundError("No file found with the name.")
    elif len(matched_files) > 1:
        raise ValueError("Multiple files found with the given name.")

    snippet_path = os.path.join(stashes_directory, matched_files[0])

    try:
        with open(snippet_path, "r") as file_stream:
            source_code = file_stream.read()

        # Call the function to copy content to clipboard
        copy_content_to_clipboard(source_code)

    except FileNotFoundError:
        print("File not found")
        raise


def copy_content_to_clipboard(content):
    # Linux
    if "linux" in sys.platform:
        # Check if xclip is installed
        if shutil.which("xclip") is None:
            print("Error: xclip not found. Install it.", file=sys.stderr)
            return
        # If xclip is installed, proceed with copying content
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=content.strip().encode(),
            check=True,
        )

    # Windows
    elif "win32" in sys.platform:
        subprocess.run(
            ["C:\\Windows\\System32\\clip.exe"], input=content.strip().encode(), check=True
        )

    # macOS
    elif "darwin" in sys.platform:
        subprocess.run(["/usr/bin/pbcopy"], input=content.strip().encode(), check=True)

    else:
        raise OSError("Unsupported operating system")