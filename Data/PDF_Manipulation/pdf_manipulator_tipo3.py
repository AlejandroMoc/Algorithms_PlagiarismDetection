from pypdf import PdfReader, PdfWriter

def rotate_pages_pdf(input_file, output_file, rotation_degrees):
    if rotation_degrees % 90 != 0:
        raise ValueError("Rotation must be a multiple of 90 degrees.")
        
    try:
        reader = PdfReader(input_file)
        writer = PdfWriter()
        
        for page in reader.pages:
            page.rotate(rotation_degrees)
            writer.add_page(page)
        
        with open(output_file, "wb") as output:
            writer.write(output)
        return True
    except Exception as error:
        print(f"An error occurred: {error}")
        return False

def crop_pages_pdf(input_file, output_file, crop_coordinates):
    if len(crop_coordinates) != 4 or not all(isinstance(coord, (int, float)) for coord in crop_coordinates):
        raise ValueError("Crop coordinates must be a tuple or list of four numeric values (left, bottom, right, top).")
    
    try:
        reader = PdfReader(input_file)
        writer = PdfWriter()
        
        for page in reader.pages:
            page.cropbox.left = crop_coordinates[0]
            page.cropbox.bottom = crop_coordinates[1]
            page.cropbox.right = crop_coordinates[2]
            page.cropbox.top = crop_coordinates[3]
            writer.add_page(page)
        
        with open(output_file, "wb") as output:
            writer.write(output)
        return True
    except Exception as error:
        print(f"An error occurred: {error}")
        return False