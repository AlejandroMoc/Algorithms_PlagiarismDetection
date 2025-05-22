import cv2
import numpy as np

def segment_image(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    ret, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)
    
    sure_background = cv2.dilate(opened_image, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(opened_image, cv2.DIST_L2, 5)
    ret, sure_foreground = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    sure_foreground = np.uint8(sure_foreground)
    unknown_region = cv2.subtract(sure_background, sure_foreground)
    
    ret, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown_region==255] = 0
    
    markers = cv2.watershed(input_image, markers)
    
    result_image = input_image.copy()
    result_image[markers == -1] = [0, 0, 255]
    
    contours, hierarchy = cv2.findContours(markers.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    return result_image