import cv2
import matplotlib.pyplot as plt

def stereo_vision(left_img, right_img):
    """
    Obtain spatial map from paired images using Enhanced Spatial Matching
    
    Args:
        left_img: Left image pair
        right_img: Right image pair
        
    Returns:
        Spatial map
    """
    # Convert images to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize matcher object
    vision = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*16,
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    # Compute disparity map
    spatial_map = vision.compute(left_gray, right_gray)
    
    # Normalize disparity map for visualization
    spatial_normalized = cv2.normalize(spatial_map, None, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return spatial_normalized

if __name__ == "__main__":
    # Load stereo images
    left_image = cv2.imread('left_image.jpg')
    right_image = cv2.imread('right_image.jpg')
    
    if left_image is None or right_image is None:
        print("Error: Could not load images")
        exit()
        
    # Get depth map
    spatial_map = stereo_vision(left_image, right_image)
    
    # Display results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    plt.title('Left Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
    plt.title('Right Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(spatial_map, cmap='hot')
    plt.title('Spatial Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()