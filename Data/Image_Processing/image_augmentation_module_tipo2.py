import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

class ImageTransformer:
    def __init__(self, picture, seed=None):
        """
        Initialize with a PIL or OpenCV picture.
        
        Parameters:
            picture (PIL.Image or np.ndarray): The picture to transform, in PIL format or as a NumPy array (BGR).
            seed (int, optional): Seed for random operations to ensure reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        if isinstance(picture, np.ndarray):
            self.picture = Image.fromarray(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
        else:
            self.picture = picture

    def twirl(self, angle_range=(-30, 30)):
        """
        Twirl the picture by a random angle within the specified range.
        
        Parameters:
            angle_range (tuple): Min and max angle for twirl.
            
        Returns:
            PIL.Image: The twirled picture.
        """
        angle = random.uniform(*angle_range)
        return self.picture.rotate(angle, resample=Image.BILINEAR)

    def mirror(self, mode='horizontal'):
        """
        Mirror the picture horizontally or vertically.
        
        Parameters:
            mode (str): 'horizontal' or 'vertical' for mirroring direction.
            
        Returns:
            PIL.Image: The mirrored picture.
        """
        if mode == 'horizontal':
            return self.picture.transpose(Image.FLIP_LEFT_RIGHT)
        elif mode == 'vertical':
            return self.picture.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError("Mode must be 'horizontal' or 'vertical'.")

    def resize_factor(self, factor_range=(0.8, 1.2)):
        """
        Resize the picture by a random factor within the specified range.
        
        Parameters:
            factor_range (tuple): Min and max scaling factor.
            
        Returns:
            PIL.Image: The resized picture.
        """
        factor = random.uniform(*factor_range)
        w, h = self.picture.size
        new_size = (int(w * factor), int(h * factor))
        return self.picture.resize(new_size, Image.BILINEAR)

    def trim(self, trim_size=(100, 100)):
        """
        Trim the picture to the specified size, randomly within the picture bounds.
        
        Parameters:
            trim_size (tuple): The width and height of the trim.
            
        Returns:
            PIL.Image: The trimmed picture.
        """
        w, h = self.picture.size
        if trim_size[0] > w or trim_size[1] > h:
            raise ValueError("Trim size must be smaller than the picture dimensions.")
        left = random.randint(0, w - trim_size[0])
        top = random.randint(0, h - trim_size[1])
        return self.picture.crop((left, top, left + trim_size[0], top + trim_size[1]))

    def adjust_tone(self, brightness=1.0, contrast=1.0, saturation=1.0):
        """
        Adjust the brightness, contrast, and saturation of the picture.
        
        Parameters:
            brightness (float): Brightness factor.
            contrast (float): Contrast factor.
            saturation (float): Saturation factor.
            
        Returns:
            PIL.Image: The tone-adjusted picture.
        """
        img = ImageEnhance.Brightness(self.picture).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        return img

    def reshape(self, size=(224, 224)):
        """
        Reshape the picture to the specified dimensions.
        
        Parameters:
            size (tuple): The target width and height.
            
        Returns:
            PIL.Image: The reshaped picture.
        """
        return self.picture.resize(size, Image.BILINEAR)

    def convert_type(self, format='JPEG'):
        """
        Change the format of the picture.
        
        Parameters:
            format (str): Desired format, e.g., 'JPEG' or 'PNG'.
            
        Returns:
            PIL.Image: The picture with updated format.
        """
        img_copy = self.picture.copy()
        img_copy.format = format
        return img_copy

    def standardize(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Standardize the picture pixel values using specified mean and standard deviation.
        
        Parameters:
            mean (tuple): Mean values for RGB channels.
            std (tuple): Standard deviation for RGB channels.
            
        Returns:
            PIL.Image: The standardized picture.
        """
        img_array = np.array(self.picture) / 255.0
        standardized_img = (img_array - mean) / std
        return Image.fromarray((standardized_img * 255).astype('uint8'))

    def to_array(self):
        """
        Convert the PIL picture to a NumPy array.
        
        Returns:
            np.ndarray: Picture as a NumPy array.
        """
        return np.array(self.picture)

    def display(self):
        """
        Show the picture.
        """
        self.picture.show()

    def add_distortion(self, distortion_level=0.1):
        """
        Add random Gaussian distortion to the picture.
        
        Parameters:
            distortion_level (float): Standard deviation of Gaussian distortion.
            
        Returns:
            PIL.Image: The distorted picture.
        """
        img_array = np.array(self.picture)
        distortion = np.random.normal(0, distortion_level * 255, img_array.shape).astype('uint8')
        distorted_img = np.clip(img_array + distortion, 0, 255).astype('uint8')
        return Image.fromarray(distorted_img)