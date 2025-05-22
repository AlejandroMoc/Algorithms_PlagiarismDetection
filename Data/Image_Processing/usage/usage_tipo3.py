from image_augmentation_module import ImageAugmentation
import cv2
import os

def load_image(image_path):
    return cv2.imread(image_path)

def check_image_loaded(image):
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

def show_augmented_image(image):
    image.show()

def save_resized_image(image, output_path):
    new_image = image.change_format('PNG')
    new_image.save(output_path)
    print(f"Resized image saved to: {output_path}")

# Load an image using OpenCV
image_path = r"Tests\image_processing\test_image.webp"
image = load_image(image_path)
check_image_loaded(image)

# Initialize the augmentation module
augmentor = ImageAugmentation(image)

# Apply various augmentations
rotated = augmentor.rotate()
flipped = augmentor.flip(mode='vertical')
scaled = augmentor.scale()
cropped = augmentor.crop((100, 100))
adjusted = augmentor.adjust_color(brightness=1.2, contrast=1.5, saturation=1.3)
resized = augmentor.resize((224, 224))
normalized = augmentor.normalize()

# Show each augmented image
show_augmented_image(rotated)
show_augmented_image(flipped)
show_augmented_image(scaled)
show_augmented_image(cropped)
show_augmented_image(adjusted)
show_augmented_image(resized)
show_augmented_image(normalized)

# Save the resized image in another format
output_dir = r"pysnippets\Image_Processing\processed_images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'resized_image.png')

save_resized_image(resized, output_path)