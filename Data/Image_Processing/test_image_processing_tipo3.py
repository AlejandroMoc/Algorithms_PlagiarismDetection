import unittest
from image_processing import resize_image, apply_filter, rotate_image

class TestImageProcessing(unittest.TestCase):
    def test_resize_image(self):
        resize_image('input_image.jpg', 'output_image_resized.jpg', (100, 100))
        # Add assertions to check if the output image is as expected

    def test_apply_filter(self):
        apply_filter('input_image.jpg', 'output_image_filtered.jpg', 'BLUR')
        # Add assertions to check if the output image is as expected

    def test_rotate_image(self):
        rotate_image('input_image.jpg', 'output_image_rotated.jpg', 90)
        # Add assertions to check if the output image is as expected

if __name__ == "__main__":
    unittest.main()