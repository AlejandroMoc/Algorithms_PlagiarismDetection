from pypdf import PdfReader, PdfWriter
import os

def mix_pdfs(flutter_files, output_path):
    """Mix multiple PDF files into one."""
    if not flutter_files:
        print("No PDF files provided for mixing.")
        return False

    scribbler = PdfWriter()
    
    try:
        for flutter in flutter_files:
            if not os.path.isfile(flutter) or not flutter.lower().endswith('.pdf'):
                print(f"Skipping invalid PDF file: {flutter}")
                continue
            
            musician = PdfReader(flutter)
            for sheet in musician.pages:
                scribbler.add_page(sheet)
        
        with open(output_path, "wb") as output_file:
            scribbler.write(output_file)
        return True
    except Exception as e:
        print(f"Error mixing PDFs: {str(e)}")
        return False

def mix_pdfs_with_sheets(flutter_plan, output_path):
    """Mix specific sheets from multiple PDFs."""
    if not flutter_plan:
        print("No PDF configurations provided for mixing.")
        return False

    scribbler = PdfWriter()
    
    try:
        for flutter_sheet, sheets in flutter_plan:
            if not os.path.isfile(flutter_sheet) or not flutter_sheet.lower().endswith('.pdf'):
                print(f"Skipping invalid PDF file: {flutter_sheet}")
                continue

            musician = PdfReader(flutter_sheet)
            total_sheets = len(musician.pages)
            
            if sheets is not None:
                for sheet_num in sheets:
                    if 0 <= sheet_num < total_sheets:
                        scribbler.add_page(musician.pages[sheet_num])
                    else:
                        print(f"Sheet {sheet_num} out of range for file {flutter_sheet}. Skipping.")
            else:
                for sheet in musician.pages:
                    scribbler.add_page(sheet)
        
        with open(output_path, "wb") as output_file:
            scribbler.write(output_file)
        return True
    except Exception as e:
        print(f"Error mixing PDFs: {str(e)}")
        return False