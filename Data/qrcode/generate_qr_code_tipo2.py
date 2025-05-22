import qrcode
from typing import Optional

def create_random_qr_code(
    information: str,
    output_path: str,
    error_correction_level: Optional[str] = "L",
    main_color: str = "black",
    background_color: str = "white"
):
    """
    Generates a QR code for the provided information and saves it as an image file.

    Args:
        information (str): The data to encode into the QR code.
        output_path (str): The file path to save the generated QR code image.
        error_correction_level (Optional[str]): Error correction level - one of 'L', 'M', 'Q', 'H'.
        main_color (str): The color to fill the QR code. Default is 'black'.
        background_color (str): The background color of the QR code. Default is 'white'.

    Example usage:
    create_random_qr_code("https://www.example.com", "random_qr.png", error_correction_level="H", main_color="blue", background_color="yellow")
    """
    
    # Map user-friendly error correction levels to qrcode.constants
    error_correction_mapping = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    
    # Set the error correction level, defaulting to 'L' if an invalid option is provided
    correction = error_correction_mapping.get(error_correction_level.upper(), qrcode.constants.ERROR_CORRECT_L)
    
    # Create a QR code instance
    qr = qrcode.QRCode(
        version=1,
        error_correction=correction,
        box_size=10,
        border=4,
    )
    
    # Add data to the QR code
    qr.add_data(information)
    qr.make(fit=True)

    # Create an image of the QR code with specified colors
    img = qr.make_image(fill_color=main_color, back_color=background_color)
    
    # Save the image
    img.save(output_path)
    print(f"QR code generated and saved as {output_path}")

if __name__ == "__main__":
    # Example usage with custom options
    create_random_qr_code(
        "https://www.example.com",
        "random_qr.png",
        error_correction_level="H",
        main_color="blue",
        background_color="yellow"
    )