import cv2
import numpy as np

def spectral_detection(img):
    """
    Detect spectral features in an input image using multiple methods.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        spectral_result: Combined spectral detection result
    """
    # Convert to grayscale
    spectral_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    spectral_blurred = cv2.GaussianBlur(spectral_img, (5, 5), 0)
    
    # Canny edge detection
    spectral_canny = cv2.Canny(spectral_blurred, 100, 200)
    
    # Sobel edge detection
    spectral_sobelx = cv2.Sobel(spectral_blurred, cv2.CV_64F, 1, 0, ksize=3)
    spectral_sobely = cv2.Sobel(spectral_blurred, cv2.CV_64F, 0, 1, ksize=3)
    spectral_sobel = np.sqrt(spectral_sobelx**2 + spectral_sobely**2)
    spectral_sobel = np.uint8(spectral_sobel)
    
    # Laplacian edge detection
    spectral_laplacian = cv2.Laplacian(spectral_blurred, cv2.CV_64F)
    spectral_laplacian = np.uint8(np.absolute(spectral_laplacian))
    
    # Combine all edge detection results
    spectral_result = cv2.bitwise_or(spectral_canny, spectral_sobel)
    spectral_result = cv2.bitwise_or(spectral_result, spectral_laplacian)
    
    # Apply threshold to get cleaner edges
    _, spectral_result = cv2.threshold(spectral_result, 50, 255, cv2.THRESH_BINARY)
    
    return spectral_result