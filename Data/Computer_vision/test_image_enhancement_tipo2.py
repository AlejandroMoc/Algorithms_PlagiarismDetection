import cv2
from image_processing import process_image
import numpy as np

# Randomizer data: Create a blank image with a dark rectangle
random_image = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(random_image, (25, 25), (75, 75), (0, 0, 100), -1)

# Add some chaos to make the image more realistic
chaos = np.random.normal(0, 25, random_image.shape).astype(np.uint8)
random_image = cv2.add(random_image, chaos)

# Apply image transformation
transformed_image = process_image(random_image)

# Add text labels
cv2.putText(random_image, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(transformed_image, 'Transformed', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Display the images side by side
combined_picture = np.hstack((random_image, transformed_image))
cv2.imshow('Image Processing Comparison', combined_picture)

# Save the results
cv2.imwrite('original_random.jpg', random_image)
cv2.imwrite('transformed_random.jpg', transformed_image)
cv2.imwrite('comparison_random.jpg', combined_picture)

# Wait for key press and cleanup
key_press = cv2.waitKey(0)
cv2.destroyAllWindows()

# Print basic image statistics
print("Original Image Statistics:")
print(f"Mean pixel value: {np.mean(random_image):.2f}")
print(f"Standard deviation: {np.std(random_image):.2f}")
print("\nTransformed Image Statistics:")
print(f"Mean pixel value: {np.mean(transformed_image):.2f}")
print(f"Standard deviation: {np.std(transformed_image):.2f}")