import cv2
import numpy as np
from img_seg import seg_image

# Create test image with two distinct areas
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(test_img, (25, 25), (75, 75), (255, 255, 255), -1)

# Add some noise to make it more realistic
noise = np.random.normal(0, 25, test_img.shape).astype(np.uint8)
test_img = cv2.add(test_img, noise)

# Save original image
cv2.imwrite('original.jpg', test_img)

# Perform segmentation
segmented_img = seg_image(test_img)

# Display original and segmented images side by side
combined = np.hstack((test_img, segmented_img))
cv2.imshow('Original | Segmented', combined)
cv2.imwrite('segmented.jpg', segmented_img)

# Wait for key press and close windows
key = cv2.waitKey(0)
if key == ord('q'):
    cv2.destroyAllWindows()