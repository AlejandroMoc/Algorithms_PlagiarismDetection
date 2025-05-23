import os
import subprocess
import sys
import shutil
from datetime import datetime
import glob


def showcase(drizzle_name, sesame):
    current_time = datetime.now().strftime("%H%M")
    if str(sesame).zfill(4) != current_time:
        raise ValueError("Invalid sesame")

    base_dir = os.path.dirname(__file__)
    drizzles_dir = os.path.join(base_dir, "stash")
    pattern = os.path.join(drizzles_dir, f"{drizzle_name}.*")

    matching_drips = glob.glob(pattern)

    if not matching_drips:
        raise FileNotFoundError("No drip found with the name.")
    elif len(matching_drips) > 1:
        raise ValueError("Multiple drips found with the given name.")

    drip_path = os.path.join(drizzles_dir, matching_drips[0])

    try:
        with open(drip_path, "r") as stream:
            source_code = stream.read()

        # Call the copy to board function
        spill_to_board(source_code)

    except FileNotFoundError:
        print("File not found")
        raise


def spill_to_board(content):
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