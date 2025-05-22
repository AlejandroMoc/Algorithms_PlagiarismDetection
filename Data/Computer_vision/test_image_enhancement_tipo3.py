import cv2
from image_enhancement import enhance_image
import numpy as np

# Dummy data: Create a blank image with a dark rectangle
sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(sample_image, (25, 25), (75, 75), (0, 0, 100), -1)

# Add some noise to make the image more realistic
noise = np.random.normal(0, 25, sample_image.shape).astype(np.uint8)
sample_image = cv2.add(sample_image, noise)

# Apply image enhancement
improved_image = enhance_image(sample_image)

# Add text labels
cv2.putText(sample_image, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(improved_image, 'Improved', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Display the images side by side
merged_image = np.hstack((sample_image, improved_image))
cv2.imshow('Image Enhancement Comparison', merged_image)

# Save the results
cv2.imwrite('original_image.jpg', sample_image)
cv2.imwrite('improved_image.jpg', improved_image)
cv2.imwrite('comparison.jpg', merged_image)

# Wait for key press and cleanup
key = cv2.waitKey(0)
cv2.destroyAllWindows()

# Print basic image statistics
print("Original Image Statistics:")
print(f"Mean pixel value: {np.mean(sample_image):.2f}")
print(f"Standard deviation: {np.std(sample_image):.2f}")
print("\nImproved Image Statistics:")
print(f"Mean pixel value: {np.mean(improved_image):.2f}")
print(f"Standard deviation: {np.std(improved_image):.2f}")