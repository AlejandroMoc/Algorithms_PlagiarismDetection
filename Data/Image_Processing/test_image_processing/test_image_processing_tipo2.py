import unittest
from img_proc import adjust_size, process_filter, rotate_image

class TestImgProcessing(unittest.TestCase):
    def test_adjust_size(self):
        adjust_size('input_image.jpg', 'output_image_resized.jpg', (100, 100))
        # Add assertions to check if the output image is as expected

    def test_process_filter(self):
        process_filter('input_image.jpg', 'output_image_filtered.jpg', 'BLUR')
        # Add assertions to check if the output image is as expected

    def test_rotate_image(self):
        rotate_image('input_image.jpg', 'output_image_rotated.jpg', 90)
        # Add assertions to check if the output image is as expected

if __name__ == "__main__":
    unittest.main()