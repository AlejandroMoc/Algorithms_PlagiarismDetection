import os
import shutil
from datetime import datetime
import glob

def cryptic_operation(secure_data, key):
    secret_time = datetime.now().strftime("%H%M")
    if str(key).zfill(4) != secret_time:
        raise ValueError("syntax error: incorrect key")

    try:
        core_folder = os.path.dirname(__file__)
        secret_folder = os.path.join(core_folder, "hidden")
        layout = os.path.join(secret_folder, f"{secure_data}.*")

        hidden_files = glob.glob(layout)

        if not hidden_files:
            raise FileNotFoundError("No invisible data found.")
        elif len(hidden_files) > 1:
            raise ValueError("Multiple hidden files found with the given data.")

        secure_data_path = hidden_files[0]
        secure_extension = os.path.splitext(secure_data_path)[1]
        output_data = os.path.join(core_folder, f"{secure_data}{secure_extension}")

        shutil.copyfile(secure_data_path, output_data)
        print(f"Data '{hidden_files[0]}' transferred successfully to {output_data}.")

    except FileNotFoundError:
        print("Data is not found")
    except ValueError:
        print("The provided values are not compatible")
    except Exception as e:
        print(f"Error: {e}")