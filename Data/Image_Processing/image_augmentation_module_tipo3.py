import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

class ImageAugmentation:
    def __init__(self, img, seed=None):
        if seed is not None:
            random.seed(seed)
        if isinstance(img, np.ndarray):
            self.img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            self.img = img

    def rotate_image(self, angle_range=(-30, 30)):
        angle = random.uniform(*angle_range)
        return self.img.rotate(angle, resample=Image.BILINEAR)

    def flip_image(self, mode='horizontal'):
        if mode == 'horizontal':
            return self.img.transpose(Image.FLIP_LEFT_RIGHT)
        elif mode == 'vertical':
            return self.img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError("Mode must be 'horizontal' or 'vertical'.")

    def scale_image(self, scale_factor=(0.8, 1.2)):
        factor = random.uniform(*scale_factor)
        w, h = self.img.size
        new_size = (int(w * factor), int(h * factor))
        return self.img.resize(new_size, Image.BILINEAR)

    def crop_image(self, crop_size=(100, 100)):
        w, h = self.img.size
        if crop_size[0] > w or crop_size[1] > h:
            raise ValueError("Crop size must be smaller than the image dimensions.")
        left = random.randint(0, w - crop_size[0])
        top = random.randint(0, h - crop_size[1])
        return self.img.crop((left, top, left + crop_size[0], top + crop_size[1]))

    def adjust_image_color(self, brightness=1.0, contrast=1.0, saturation=1.0):
        img = ImageEnhance.Brightness(self.img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        return img

    def resize_image(self, size=(224, 224)):
        return self.img.resize(size, Image.BILINEAR)

    def change_image_format(self, img_format='JPEG'):
        img_copy = self.img.copy()
        img_copy.format = img_format
        return img_copy

    def normalize_image(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        img_array = np.array(self.img) / 255.0
        normalized_img = (img_array - mean) / std
        return Image.fromarray((normalized_img * 255).astype('uint8'))

    def to_numpy_array(self):
        return np.array(self.img)

    def display_image(self):
        self.img.show()

    def add_noise_to_image(self, noise_level=0.1):
        img_array = np.array(self.img)
        noise = np.random.normal(0, noise_level * 255, img_array.shape).astype('uint8')
        noisy_img = np.clip(img_array + noise, 0, 255).astype('uint8')
        return Image.fromarray(noisy_img)