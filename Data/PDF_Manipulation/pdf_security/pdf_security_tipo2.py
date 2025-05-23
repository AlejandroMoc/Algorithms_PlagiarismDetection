from pypdf import PdfReader, PdfWriter
import os

def secure_document(entrance_path, exit_path, user_key, admin_key):
    """Protect a document with user and admin keys."""
    if not os.path.isfile(entrance_path) or not entrance_path.lower().endswith('.pdf'):
        print(f"Invalid entrance document: {entrance_path}")
        return False
    
    try:
        reader = PdfReader(entrance_path)
        writer = PdfWriter()
        
        for leaf in reader.pages:
            writer.add_page(leaf)
        
        # Protect the document
        writer.encrypt(user_password=user_key, owner_password=admin_key, use_128bit=True)
        
        with open(exit_path, "wb") as exit_file:
            writer.write(exit_file)
        print("Document protected successfully.")
        return True
    except Exception as e:
        print(f"Error protecting document: {str(e)}")
        return False

def unprotect_document(entrance_path, exit_path, key):
    """Unlock a document using the provided key."""
    if not os.path.isfile(entrance_path) or not entrance_path.lower().endswith('.pdf'):
        print(f"Invalid entrance document: {entrance_path}")
        return False
    
    try:
        reader = PdfReader(entrance_path)
        
        # Check if the document is encrypted
        if reader.is_encrypted:
            try:
                if reader.decrypt(key) != 1:
                    print("Incorrect key. Unlock failed.")
                    return False
            except Exception as e:
                print(f"Unlock failed: {str(e)}")
                return False
        else:
            print("Document is not encrypted.")
            return False
        
        writer = PdfWriter()
        for leaf in reader.pages:
            writer.add_page(leaf)
        
        with open(exit_path, "wb") as exit_file:
            writer.write(exit_file)
        print("Document unlocked successfully.")
        return True
    except Exception as e:
        print(f"Error unlocking document: {str(e)}")
        return False