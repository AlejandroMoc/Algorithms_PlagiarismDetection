from img_aug_module import ImgAug
import cv2
import os

# Load an image using OpenCV
path_to_img = r"Tests\img_proc\test_img.webp"
img = cv2.imread(path_to_img)

# Check if image was loaded successfully
if img is None:
    raise FileNotFoundError(f"Image not found at path: {path_to_img}")

# Initialize the augmentation module
aug = ImgAug(img)

# Apply various augmentations
rotated_img = aug.rotate()
flipped_img = aug.flip(mode='vertical')
scaled_img = aug.scale()
cropped_img = aug.crop((100, 100))
adjusted_img = aug.adjust_color(brightness=1.2, contrast=1.5, saturation=1.3)
resized_img = aug.resize((224, 224))
normalized_img = aug.normalize()

# Show each augmented image
rotated_img.show()
flipped_img.show()
scaled_img.show()
cropped_img.show()
adjusted_img.show()
resized_img.show()
normalized_img.show()

# Save the resized image in another format
output_directory = r"pysnippets\Img_Processing\processed_imgs"
os.makedirs(output_directory, exist_ok=True)
output_path = os.path.join(output_directory, 'resized_img.png')

new_img = aug.change_format('PNG')
new_img.save(output_path)

print(f"Resized image saved to: {output_path}")