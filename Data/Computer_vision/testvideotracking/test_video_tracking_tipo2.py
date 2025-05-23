import cv2
def detect_anomaly_in_footage(footage_path):
    # Initialize video capture
    clip = cv2.VideoCapture(footage_path)
    
    # Read first frame
    ret, image = clip.read()
    if not ret:
        print("Failed to read video")
        return
        
    # Select region of interest for tracking
    region = cv2.selectROI("Select Anomaly", image, False)
    cv2.destroyWindow("Select Anomaly")
    
    # Initialize anomaly detector
    anomaly_detector = cv2.TrackerCSRT_create()
    anomaly_detector.init(image, region)
    
    while True:
        ret, image = clip.read()
        if not ret:
            break
            
        # Update anomaly detector
        success, box = anomaly_detector.update(image)
        
        if success:
            # Draw detection box
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # Display detection result
        cv2.imshow("Anomaly Detection", image)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    clip.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your video path
    footage_path = "sample_video.mp4"
    detect_anomaly_in_footage(footage_path)