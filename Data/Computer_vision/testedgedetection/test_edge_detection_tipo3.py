import cv2
import numpy as np
from edge_detection import detect_edges

# Load a real test image
img_test = cv2.imread('test_image.jpg')
if img_test is None:
    # If no image found, create dummy data
    img_test = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Convert to grayscale
gray_img = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Detect edges using imported function
edges_img = detect_edges(blurred_img)

# Display original and edges side by side
cv2.imshow('Original', img_test)
cv2.imshow('Edges', edges_img)

# Wait for key press and close windows
key_pressed = cv2.waitKey(0)
if key_pressed == ord('q') or key_pressed == 27:  # Exit on 'q' or ESC
    cv2.destroyAllWindows()