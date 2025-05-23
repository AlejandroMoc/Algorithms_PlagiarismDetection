# scan_barcode.py

import cv2
from pyzbar.pyzbar import decode
from typing import List, Optional

def scan_barcode(image_path: str) -> Optional[List[str]]:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    barcodes = decode(img)

    if barcodes:
        decoded_data = []
        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            decoded_data.append(barcode_data)
            print(f"Barcode data: {barcode_data}")
        return decoded_data
    else:
        print("No barcode found")
        return None

if __name__ == "__main__":
    scan_barcode("example_barcode.png")