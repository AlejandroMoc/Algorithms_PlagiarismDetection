from pypdf import PdfReader
import os

def extract_text_from_pdf(filepath):
    if not os.path.isfile(filepath) or not filepath.lower().endswith('.pdf'):
        print(f"Invalid file path: {filepath}")
        return None

    try:
        reader = PdfReader(filepath)
        text_content = []

        page_index = 0
        while page_index < len(reader.pages):
            page_text = reader.pages[page_index].extract_text()
            if page_text:
                text_content.append(page_text)
            else:
                text_content.append("[No extractable text]")

            page_index += 1

        return {
            'num_pages': len(reader.pages),
            'content': text_content
        }
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def extract_metadata_from_pdf(filepath):
    if not os.path.isfile(filepath) or not filepath.lower().endswith('.pdf'):
        print(f"Invalid file path: {filepath}")
        return None

    try:
        reader = PdfReader(filepath)
        metadata_keys = ['/Author', '/Creator', '/Producer', '/Subject', '/Title']
        
        metadata = {}
        key_index = 0
        while key_index < len(metadata_keys):
            key = metadata_keys[key_index]
            metadata[key] = reader.metadata.get(key, "Unknown")
            key_index += 1

        return metadata
    except Exception as e:
        print(f"Error reading PDF metadata: {str(e)}")
        return None