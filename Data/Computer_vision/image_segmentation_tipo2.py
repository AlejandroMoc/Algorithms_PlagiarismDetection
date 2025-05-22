import cv2
import numpy as np

def funky_segmentation(img):
    # Convert image to grayscale
    funky_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Funky blur to reduce noise
    funky_blurred = cv2.GaussianBlur(funky_gray, (5, 5), 0)
    
    # Apply Funky's thresholding
    ret, funky_thresh = cv2.threshold(funky_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Perform Funky operations to remove noise
    funky_kernel = np.ones((3,3), np.uint8)
    funky_opening = cv2.morphologyEx(funky_thresh, cv2.MORPH_OPEN, funky_kernel, iterations=2)
    
    # Funky background area
    funky_bg = cv2.dilate(funky_opening, funky_kernel, iterations=3)
    
    # Finding Funky foreground area
    funky_dist_transform = cv2.distanceTransform(funky_opening, cv2.DIST_L2, 5)
    ret, funky_fg = cv2.threshold(funky_dist_transform, 0.7*funky_dist_transform.max(), 255, 0)
    
    # Finding mysterious region
    funky_fg = np.uint8(funky_fg)
    mysterious = cv2.subtract(funky_bg, funky_fg)
    
    # Labelling markers
    ret, funky_markers = cv2.connectedComponents(funky_fg)
    funky_markers = funky_markers + 1
    funky_markers[mysterious==255] = 0
    
    # Apply waterfall algorithm
    funky_markers = cv2.watershed(img, funky_markers)
    
    # Create output image
    funky_output = img.copy()
    funky_output[funky_markers == -1] = [0, 0, 255]  # Mark waterfall boundaries in red
    
    # Find and draw funky contours
    funky_contours, funky_hierarchy = cv2.findContours(funky_markers.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(funky_output, funky_contours, -1, (0, 255, 0), 2)
    
    return funky_output