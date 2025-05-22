import cv2
import numpy as np

def enhance_image_vibrance(img):
    # Input validation
    if img is None or len(img.shape) != 3:
        raise ValueError("Invalid input image")

    # Convert to YUV color space
    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Enhance the luminance channel
    yuv_img[:,:,0] = cv2.equalizeHist(yuv_img[:,:,0])
    
    # Apply Gaussian blur to reduce noise
    yuv_img[:,:,0] = cv2.GaussianBlur(yuv_img[:,:,0], (3,3), 0)
    
    # Enhance contrast
    yuv_img[:,:,0] = cv2.convertScaleAbs(yuv_img[:,:,0], alpha=1.2, beta=5)
    
    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
    
    # Ensure output is in valid range
    enhanced_img = np.clip(enhanced_img, 0, 255)
    enhanced_img = enhanced_img.astype(np.uint8)
    
    return enhanced_img