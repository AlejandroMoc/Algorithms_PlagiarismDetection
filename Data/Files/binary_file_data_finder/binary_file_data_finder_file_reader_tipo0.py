python
# file_reader_updated.py

def get_file_contents(file_path):
    try:
        with open(file_path, "r") as file_obj:
            return file_obj.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except IOError as err:
        print(f"Error reading file '{file_path}': {err}")
        return None

if __name__ == "__main__":
    file_contents = get_file_contents("example.txt")
    if file_contents is not None:
        print(file_contents)