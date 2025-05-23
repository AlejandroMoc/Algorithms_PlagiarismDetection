import cv2
import numpy as np

def match_features(image_path1, image_path2):
    # Read images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    
    # Convert to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift_detector = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    keypoints1, descriptors1 = sift_detector.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = sift_detector.detectAndCompute(gray_image2, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_parameters = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_parameters = dict(checks=50)
    
    # FLANN matcher
    flann_matcher = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    matches = flann_matcher.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match1, match2 in matches:
        if match1.distance < 0.7 * match2.distance:
            good_matches.append(match1)
    
    # Draw matches
    result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return result_image

if __name__ == "__main__":
    # Example usage
    image_path1 = "image1.jpg"
    image_path2 = "image2.jpg"
    
    result_image = match_features(image_path1, image_path2)
    cv2.imshow("Feature Matches", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()