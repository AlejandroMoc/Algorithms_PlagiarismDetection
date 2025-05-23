import cv2
import numpy as np

def generate_magic_flow(first_image, second_image):
    # Convert frames to grayscale
    first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using Farneback method
    magic = cv2.calcOpticalFlowFarneback(first_gray, second_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Create visualization
    power, direction = cv2.cartToPolar(magic[..., 0], magic[..., 1])
    
    # Create HSV image for visualization
    rainbow = np.zeros_like(first_image)
    rainbow[..., 1] = 255  # Saturation
    
    # Use direction for hue and power for value
    rainbow[..., 0] = direction * 180 / np.pi / 2
    rainbow[..., 2] = cv2.normalize(power, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert HSV to BGR for visualization
    magic_visualization = cv2.cvtColor(rainbow, cv2.COLOR_HSV2BGR)
    
    return magic, magic_visualization