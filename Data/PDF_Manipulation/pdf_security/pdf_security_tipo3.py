from pypdf import PdfReader, PdfWriter
import os

def encrypt_pdf_file(input_file_path, output_file_path, user_password, owner_password):
    if not os.path.isfile(input_file_path) or not input_file_path.lower().endswith('.pdf'):
        print(f"Invalid input file: {input_file_path}")
        return False
    
    try:
        reader = PdfReader(input_file_path)
        writer = PdfWriter()
        
        for page in reader.pages:
            writer.add_page(page)
        
        writer.encrypt(user_password=user_password, owner_password=owner_password, use_128bit=True)
        
        with open(output_file_path, "wb") as output_file:
            writer.write(output_file)
        print("PDF encrypted successfully.")
        return True
    except Exception as e:
        print(f"Error encrypting PDF: {str(e)}")
        return False

def decrypt_pdf_file(input_file_path, output_file_path, password):
    if not os.path.isfile(input_file_path) or not input_file_path.lower().endswith('.pdf'):
        print(f"Invalid input file: {input_file_path}")
        return False
    
    try:
        reader = PdfReader(input_file_path)
        
        if reader.is_encrypted:
            try:
                if reader.decrypt(password) != 1:
                    print("Incorrect password. Decryption failed.")
                    return False
            except Exception as e:
                print(f"Decryption failed: {str(e)}")
                return False
        else:
            print("PDF is not encrypted.")
            return False
        
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        
        with open(output_file_path, "wb") as output_file:
            writer.write(output_file)
        print("PDF decrypted successfully.")
        return True
    except Exception as e:
        print(f"Error decrypting PDF: {str(e)}")
        return False