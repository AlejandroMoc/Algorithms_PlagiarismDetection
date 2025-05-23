# file_sorter.py

import os
import shutil
import sys
import logging

logging.basicConfig(level=logging.INFO)

def sort_files_by_category(folder_path):
    """
    Sorts files in the specified folder by file category, moving them into categorized subfolders.

    Args:
        folder_path (str): The path to the folder to sort.

    Returns:
        None

    Raises:
        ValueError: If the folder_path does not exist.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder path '{folder_path}' does not exist.")
    
    # Define file clusters and their extensions
    file_clusters = {
        'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.svg', '.webp'],
        'Documents': ['.pdf', '.docx', '.doc', '.txt', '.xls', '.xlsx', '.pptx', '.ppt', '.csv'],
        'Videos': ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv'],
        'Music': ['.mp3', '.wav', '.flac', '.aac', '.ogg'],
        'Archives': ['.zip', '.rar', '.tar', '.gz', '.7z'],
        'Scripts': ['.py', '.js', '.html', '.css', '.c', '.cpp'],
        'Fonts': ['.ttf', '.otf', '.woff', '.woff2'],
        'System': ['.exe', '.dll', '.sys', '.bin']
    }

    unsorted_folder = os.path.join(folder_path, "Unsorted")
    os.makedirs(unsorted_folder, exist_ok=True)

    # Sort files
    for fname in os.listdir(folder_path):
        file_path = os.path.join(folder_path, fname)

        # Skip folders
        if os.path.isdir(file_path):
            continue

        # Get file extension
        _, file_ext = os.path.splitext(fname)
        file_ext = file_ext.lower()

        # Move the file to the appropriate cluster folder
        moved = False  # Flag to check if the file was moved
        for cluster, exts in file_clusters.items():
            if file_ext in exts:
                cluster_path = os.path.join(folder_path, cluster)
                os.makedirs(cluster_path, exist_ok=True)
                try:
                    shutil.move(file_path, os.path.join(cluster_path, fname))
                    logging.info(f"Moved: {fname} -> {cluster}")
                except Exception as e:
                    logging.error(f"Error moving file {fname}: {e}")
                moved = True
                break

        # Move unrecognized file to 'Unsorted'
        if not moved:
            try:
                shutil.move(file_path, os.path.join(unsorted_folder, fname))
                logging.warning(f"Unrecognized file type for: {fname}. Moved to 'Unsorted'.")
            except Exception as e:
                logging.error(f"Error moving unrecognized file {fname}: {e}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder_to_sort = sys.argv[1]
    else:
        folder_to_sort = input("Enter the path to the folder you want to sort: ")

    try:
        sort_files_by_category(folder_to_sort)
    except ValueError as e:
        logging.error(e)