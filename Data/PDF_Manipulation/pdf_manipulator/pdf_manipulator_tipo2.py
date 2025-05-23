from pypdf import PdfReader, PdfWriter

def sizzle_pdf_folios(input_doc, output_doc, rotation_angle):
    """Rotate all pages in a PDF by specified degrees."""
    if rotation_angle % 90 != 0:
        raise ValueError("Rotation must be a multiple of 90 degrees.")
        
    try:
        reader = PdfReader(input_doc)
        writer = PdfWriter()
        
        for folio in reader.pages:
            folio.rotate(rotation_angle)
            writer.add_page(folio)
        
        with open(output_doc, "wb") as result_file:
            writer.write(result_file)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def trim_pdf_folios(input_doc, output_doc, trimming_box):
    """Crop all pages in a PDF using specified coordinates."""
    if len(trimming_box) != 4 or not all(isinstance(coord, (int, float)) for coord in trimming_box):
        raise ValueError("Trimming box must be a tuple or list of four numeric values (left, bottom, right, top).")
    
    try:
        reader = PdfReader(input_doc)
        writer = PdfWriter()
        
        for folio in reader.pages:
            folio.cropbox.left = trimming_box[0]
            folio.cropbox.bottom = trimming_box[1]
            folio.cropbox.right = trimming_box[2]
            folio.cropbox.top = trimming_box[3]
            writer.add_page(folio)
        
        with open(output_doc, "wb") as result_file:
            writer.write(result_file)
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False