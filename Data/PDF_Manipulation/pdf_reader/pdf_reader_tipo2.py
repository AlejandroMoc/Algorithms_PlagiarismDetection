from pypdf import PdfReader
import os

def extract_text_from_document(file_path):
    """Read and extract text from a PDF file."""
    if not os.path.isfile(file_path) or not file_path.lower().endswith('.pdf'):
        print(f"Invalid file path: {file_path}")
        return None

    try:
        doc_reader = PdfReader(file_path)
        extracted_text = []

        for page in doc_reader.pages:
            page_text = page.extract_text()
            extracted_text.append(page_text if page_text else "[No extractable text]")

        return {
            'num_pages': len(doc_reader.pages),
            'text_content': extracted_text
        }
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def extract_metadata_from_document(file_path):
    """Extract metadata from a PDF file."""
    if not os.path.isfile(file_path) or not file_path.lower().endswith('.pdf'):
        print(f"Invalid file path: {file_path}")
        return None

    try:
        doc_reader = PdfReader(file_path)
        metadata_keys = ['/Author', '/Creator', '/Producer', '/Subject', '/Title']
        
        # Use .get() directly on metadata to simplify handling missing values
        metadata = {key: doc_reader.metadata.get(key, "Unknown") for key in metadata_keys}
        return metadata
    except Exception as e:
        print(f"Error reading PDF metadata: {str(e)}")
        return None