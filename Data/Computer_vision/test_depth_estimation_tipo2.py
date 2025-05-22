import cv2
import matplotlib.pyplot as plt
from depth_estimation import estimate_depth

def analyze_depth():
    # Load stereo image pair
    pic1 = cv2.imread('data/left.jpg')
    pic2 = cv2.imread('data/right.jpg')
    
    # Convert to grayscale
    gray_pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
    
    # Estimate depth map
    depth_map_result = estimate_depth(gray_pic1, gray_pic2)
    
    # Visualize results
    plt.figure(figsize=(12,5))
    
    plt.subplot(131)
    plt.imshow(pic1[...,::-1])
    plt.title('Left Image')
    plt.axis('off')
    
    plt.subplot(132) 
    plt.imshow(pic2[...,::-1])
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
    analyze_depth()