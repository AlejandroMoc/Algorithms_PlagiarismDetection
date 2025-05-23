import cv2
import numpy as np
from edge_detection import detect_edges

# Load a real test image
sample_img = cv2.imread('test_image.jpg')
if sample_img is None:
    # If no image found, create dummy data
    sample_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Convert to grayscale
gray_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# Detect edges using imported function
edge_img = detect_edges(blurred_img)

# Display original and edges side by side
cv2.imshow('Original', sample_img)
cv2.imshow('Edges', edge_img)

# Wait for key press and close windows
key_press = cv2.waitKey(0)
if key_press == ord('q') or key_press == 27:  # Exit on 'q' or ESC
    cv2.destroyAllWindows()