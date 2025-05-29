import cv2

def detect_anomaly_in_motion(video_path):
    # Initialize video capture
    movie = cv2.VideoCapture(video_path)
    
    # Read first frame
    ret, frame = movie.read()
    if not ret:
        print("Failed to read video")
        return
    
    # Select ROI (Region of Interest)
    region = cv2.selectROI(frame)
    
    # Initialize tracker
    anomaly_detector = cv2.TrackerCSRT_create()
    anomaly_detector.init(frame, region)
    
    while True:
        ret, frame = movie.read()
        if not ret:
            break
        
        # Update tracker
        success, box = anomaly_detector.update(frame)
        
        if success:
            # Draw bounding box
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display result
        cv2.imshow('Anomaly Detection', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    movie.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    video_file = "sample_video.mp4"  # Replace with your video file
    detect_anomaly_in_motion(video_file)