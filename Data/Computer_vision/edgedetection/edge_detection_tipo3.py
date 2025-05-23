import cv2
import numpy as np

def detect_edges(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    canny_edges = cv2.Canny(blurred_image, 100, 200)
    
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_edges = np.uint8(sobel_edges)
    
    laplacian_edges = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplacian_edges = np.uint8(np.absolute(laplacian_edges))
    
    combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
    combined_edges = cv2.bitwise_or(combined_edges, laplacian_edges)
    
    _, cleaned_edges = cv2.threshold(combined_edges, 50, 255, cv2.THRESH_BINARY)
    
    return cleaned_edges