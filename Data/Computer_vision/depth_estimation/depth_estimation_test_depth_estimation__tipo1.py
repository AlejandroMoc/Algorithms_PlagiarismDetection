import cv2

def funky_finding(pic1_path, pic2_path):
    # Read images
    pic1 = cv2.imread(pic1_path)
    pic2 = cv2.imread(pic2_path)
    
    # Convert to grayscale
    gray_pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sifter = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp_pic1, des_pic1 = sifter.detectAndCompute(gray_pic1, None)
    kp_pic2, des_pic2 = sifter.detectAndCompute(gray_pic2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_pic1, des_pic2, k=2)

    # Apply ratio test
    good_matches = []
    for match1, match2 in matches:
        if match1.distance < 0.7 * match2.distance:
            good_matches.append(match1)

    # Draw matches
    pic_matches = cv2.drawMatches(pic1, kp_pic1, pic2, kp_pic2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return pic_matches

if __name__ == "__main__":
    # Example usage
    pic1_path = "image1.jpg"
    pic2_path = "image2.jpg"
    
    result = funky_finding(pic1_path, pic2_path)
    
    # Display result
    cv2.imshow("Funky Matches", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()