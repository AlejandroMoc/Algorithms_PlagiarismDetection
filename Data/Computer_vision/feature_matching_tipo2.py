import cv2
import numpy as np

def funky_feature(img1_path, img2_path):
    # Read images
    funky_img1 = cv2.imread(img1_path)
    funky_img2 = cv2.imread(img2_path)
    
    # Convert to grayscale
    funky_gray1 = cv2.cvtColor(funky_img1, cv2.COLOR_BGR2GRAY)
    funky_gray2 = cv2.cvtColor(funky_img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    funky_sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    funky_kp1, funky_des1 = funky_sift.detectAndCompute(funky_gray1, None)
    funky_kp2, funky_des2 = funky_sift.detectAndCompute(funky_gray2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    funky_index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    funky_search_params = dict(checks=50)
    
    # FLANN matcher
    funky_flann = cv2.FlannBasedMatcher(funky_index_params, funky_search_params)
    funky_matches = funky_flann.knnMatch(funky_des1, funky_des2, k=2)
    
    # Apply ratio test
    good_funky_matches = []
    for funky_m, funky_n in funky_matches:
        if funky_m.distance < 0.7 * funky_n.distance:
            good_funky_matches.append(funky_m)
    
    # Draw matches
    funky_result = cv2.drawMatches(funky_img1, funky_kp1, funky_img2, funky_kp2, good_funky_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return funky_result

if __name__ == "__main__":
    # Example usage
    funky_img1_path = "image1.jpg"
    funky_img2_path = "image2.jpg"
    
    funky_result = funky_feature(funky_img1_path, funky_img2_path)
    cv2.imshow("Funky Feature Matches", funky_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()