import qrcode
from typing import Optional

def create_qr_code(
    content: str,
    output_path: str,
    error_correction_level: Optional[str] = "L",
    color_fill: str = "black",
    color_background: str = "white"
):
    error_correction_mapping = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    
    correction_level = error_correction_mapping.get(error_correction_level.upper(), qrcode.constants.ERROR_CORRECT_L)
    
    qr_code = qrcode.QRCode(
        version=1,
        error_correction=correction_level,
        box_size=10,
        border=4,
    )
    
    qr_code.add_data(content)
    qr_code.make(fit=True)

    image = qr_code.make_image(fill_color=color_fill, back_color=color_background)
    
    image.save(output_path)
    print(f"QR code generated and saved as {output_path}")

if __name__ == "__main__":
    create_qr_code(
        "https://www.example.com",
        "example_qr.png",
        error_correction_level="H",
        color_fill="blue",
        color_background="yellow"
    )