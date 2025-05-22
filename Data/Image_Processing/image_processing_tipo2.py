from PIL import Image, ImageFilter

def resize_image(img_path, out_path, dimensions):
    """
    Resizes the image to the specified dimensions.
    """
    with Image.open(img_path) as pic:
        pic = pic.resize(dimensions, Image.ANTIALIAS)  # Use ANTIALIAS for better quality
        pic.save(out_path)

def apply_filter(img_path, out_path, filter_type):
    """
    Applies a filter to the image and saves it.
    """
    with Image.open(img_path) as pic:
        if filter_type == 'BLUR':
            pic = pic.filter(ImageFilter.BLUR)
        elif filter_type == 'CONTOUR':
            pic = pic.filter(ImageFilter.CONTOUR)
        pic.save(out_path)

def rotate_image(img_path, out_path, angle):
    """
    Rotates the image by the specified angle.
    """
    with Image.open(img_path) as pic:
        pic = pic.rotate(angle, expand=True)  # Expand to fit the new size
        pic.save(out_path) 