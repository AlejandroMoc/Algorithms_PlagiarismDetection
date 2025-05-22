import urllib.request
import re
import html
import subprocess
import sys
import shutil


def copy_text_to_clipboard(text):
    if "linux" in sys.platform:
        if shutil.which("xclip") is None:
            print("Error: xclip not found. Install it.", file=sys.stderr)
            return
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=text.strip().encode(),
            check=True,
        )
    elif "win32" in sys.platform:
        subprocess.run(
            ["C:\\Windows\\System32\\clip.exe"], input=text.strip().encode(), check=True
        )
    elif "darwin" in sys.platform:
        subprocess.run(["/usr/bin/pbcopy"], input=text.strip().encode(), check=True)
    else:
        raise OSError("Unsupported operating system")


def fetch_content(url_name):
    url = f"https://cl1p.net/{url_name}"
    try:
        with urllib.request.urlopen(url) as response:
            html_content = response.read().decode("utf-8")
        match = re.search(
            r'<textarea[^>]*id="cl1pTextArea"[^>]*>(.*?)</textarea>',
            html_content,
            re.DOTALL,
        )
        if match:
            content = html.unescape(match.group(1)).strip()
            if content:
                return content
    except (urllib.error.HTTPError, urllib.error.URLError):
        pass
    return content


def display_content(url_name):
    try:
        content = fetch_content(url_name)
        if content:
            print("The content is: ", content)
        else:
            print(
                "Nothing found. The clipboard might be empty or you have entered a wrong URL."
            )
    except Exception as e:
        print(f"Error occurred: {e}")


def copy_content(url_name):
    try:
        content = fetch_content(url_name)
        if content:
            copy_text_to_clipboard(content)
            print("Content copied to clipboard.")
        else:
            print(
                "Nothing found. The clipboard might be empty or you have entered a wrong URL."
            )
    except Exception as e:
        print(f"Error occurred: {e}")


def save_to_file(url_name, file_name):
    try:
        content = fetch_content(url_name)
        with open(file_name, "w") as file:
            file.write(content)
        print("Content written to file successfully.")
    except IOError as e:
        print(f"Error writing to file: {e}")