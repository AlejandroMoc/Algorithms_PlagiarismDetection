# decode_image_data.py

import cv2
from pyzbar.pyzbar import decode
from typing import List, Optional

def decode_image_data(image_path: str) -> Optional[List[str]]:
    """
    Decodes image data from the provided image file and returns the decoded data.

    Args:
        image_path (str): The file path of the image containing the encoded data.

    Returns:
        Optional[List[str]]: A list of decoded data from the image, or None if no data is found.

    Example usage:
    decode_image_data("example_image.png") -> ["https://www.example.com"]
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    encoded_data = decode(img)

    if encoded_data:
        decoded_data = []
        for data in encoded_data:
            decoded_item = data.data.decode('utf-8')
            decoded_data.append(decoded_item)
            print(f"Decoded data: {decoded_item}")
        return decoded_data
    else:
        print("No data found in image")
        return None

if __name__ == "__main__":
    # Example usage
    decode_image_data("example_image.png")