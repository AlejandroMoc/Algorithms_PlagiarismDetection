import cv2
import matplotlib.pyplot as plt
from depth_estimation import estimate_depth

def validate_depth_estimation():
    # Load stereo image pair
    left_image = cv2.imread('data/left.jpg')
    right_image = cv2.imread('data/right.jpg')
    
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    
    # Estimate depth map
    depth_map_result = estimate_depth(left_gray, right_gray)
    
    # Visualize results
    plt.figure(figsize=(12,5))
    
    plt.subplot(131)
    plt.imshow(left_image[...,::-1])
    plt.title('Left Image')
    plt.axis('off')
    
    plt.subplot(132) 
    plt.imshow(right_image[...,::-1])
    plt.title('Right Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(depth_map_result, cmap='plasma')
    plt.title('Estimated Depth Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    validate_depth_estimation()