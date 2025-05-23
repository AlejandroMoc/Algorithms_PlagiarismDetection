# file_manager.py

import os
import shutil
import sys
import logging

logging.basicConfig(level=logging.INFO)

def sort_files_by_type(folder_directory):
    if not os.path.exists(folder_directory):
        raise ValueError(f"The folder directory '{folder_directory}' does not exist.")
    
    file_categories = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.svg', '.webp'],
        'Documents': ['.pdf', '.docx', '.doc', '.txt', '.xls', '.xlsx', '.pptx', '.ppt', '.csv'],
        'Videos': ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv'],
        'Music': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'Archives': ['.zip', '.rar', '.tar', '.gz', '.7z'],
        'Scripts': ['.py', '.js', '.html', '.css', '.c', '.cpp'],
        'Fonts': ['.ttf', '.otf', '.woff', '.woff2'],
        'System': ['.exe', '.dll', '.sys', '.bin']
    }

    uncategorized_folder = os.path.join(folder_directory, "Uncategorized")
    os.makedirs(uncategorized_folder, exist_ok=True)

    for file_name in os.listdir(folder_directory):
        file_path = os.path.join(folder_directory, file_name)

        if os.path.isdir(file_path):
            continue

        _, file_extension = os.path.splitext(file_name)
        file_extension = file_extension.lower()

        moved = False
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                category_path = os.path.join(folder_directory, category)
                os.makedirs(category_path, exist_ok=True)
                try:
                    shutil.move(file_path, os.path.join(category_path, file_name))
                    logging.info(f"Moved: {file_name} -> {category}")
                except Exception as error:
                    logging.error(f"Error moving file {file_name}: {error}")
                moved = True
                break

        if not moved:
            try:
                shutil.move(file_path, os.path.join(uncategorized_folder, file_name))
                logging.warning(f"Unrecognized file type for: {file_name}. Moved to 'Uncategorized'.")
            except Exception as error:
                logging.error(f"Error moving unrecognized file {file_name}: {error}")

def main():
    if len(sys.argv) > 1:
        folder_to_sort = sys.argv[1]
    else:
        folder_to_sort = input("Enter the path to the folder you want to organize: ")

    try:
        sort_files_by_type(folder_to_sort)
    except ValueError as error:
        logging.error(error)

if __name__ == "__main__":
    main()