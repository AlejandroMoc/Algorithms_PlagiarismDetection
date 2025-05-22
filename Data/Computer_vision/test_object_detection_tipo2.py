import cv2
import numpy as np
from object_detection import detect_objects

# Randomizer: Create an image with random shapes
sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(sample_image, (20, 20), (80, 80), (0, 255, 0), -1)  # Simulated object

# Identify shapes in the image
recognized_shapes = detect_objects(sample_image)

# Outline shapes with bounding boxes
result_image = sample_image.copy()
if recognized_shapes:
    for (a, b, c, d) in recognized_shapes:
        # Draw box around recognized shape
        cv2.rectangle(result_image, (a, b), (a + c, b + d), (0, 0, 255), 2)
        # Add identifier
        cv2.putText(result_image, 'Shape', (a, b-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Show the original and identified shapes
cv2.imshow('Original Image', sample_image)
cv2.imshow('Shape Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()