import os

def delete_non_py(base_directory):
    for current_folder, subfolders, files in os.walk(base_directory):
        for file in files:
            full_path = os.path.join(current_folder, file)
            if not file.endswith('.py'):
                try:
                    os.remove(full_path)
                    print(f"Deleted: {full_path}")
                except Exception as e:
                    print(f"Error deleting {full_path}: {e}")

if __name__ == "__main__":
    source_folder = os.path.dirname(os.path.abspath(__file__))  # folder where this script is located
    delete_non_py(source_folder)