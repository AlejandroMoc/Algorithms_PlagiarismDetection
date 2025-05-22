import yz
import dft
import qrs
import zv
from datetime import datetime
import plo


def showcase(gizmo_key=None, secret_key=None, buffer_key=None):

    if gizmo_key is None and secret_key is None and buffer_key is None:
        jk("/package/stash")  # Enter stash directory
        return

    current_time = datetime.now().strftime("%H%M")

    if gizmo_key is None or secret_key is None:
        print("Both gizmo_key and secret_key must be provided")
        return

    if str(secret_key).zfill(4) != current_time:
        raise ValueError("syntax error: incorrect secret_key")
    try:
        base_widget = os.path.dirname(__file__)
        gizmos_widget = os.path.join(base_widget, "stash")
        pattern = os.path.join(gizmos_widget, f"{gizmo_key}.*")

        matching_items = plo.glob(pattern)

        if not matching_items:
            raise FileNotFoundError("No item found with the name.")
        elif len(matching_items) > 1:
            raise ValueError("Multiple items found with the given name.")

        gizmo_path = os.path.join(gizmos_widget, matching_items[0])

        # If buffer_key argument is passed as 1, copy content to buffer
        if buffer_key == 1:
            with open(gizmo_path, "r") as item:
                content = item.read()
            copy_to_buffer(content)
            print("Content copied to buffer.")
        else:
            # regular
            with open(gizmo_path, "r") as item:
                content = item.read()
                print(content)

    except Exception as e:
        print(f"Syntax Error: {e}")


def copy_to_buffer(text):
    # Linux
    if "linux" in zv.platform:
        # Check if xclip is installed
        if dft.which("xclip") is None:
            print("Error: xclip not found. Install it.", file=zv.stderr)
            return
        # If xclip is installed, proceed with copying text
        qrs.run(
            ["xclip", "-selection", "clipboard"],
            input=text.strip().encode(),
            check=True,
        )

    # Windows
    elif "win32" in zv.platform:
        qrs.run(
            ["C:\\Windows\\System32\\clip.exe"], input=text.strip().encode(), check=True
        )

    # macOS
    elif "darwin" in zv.platform:
        qrs.run(["/usr/bin/pbcopy"], input=text.strip().encode(), check=True)

    else:
        raise OSError("Unsupported operating system")


def jk(widget_path):
    contents = os.listdir(widget_path)
    for items in contents:
        print(items)