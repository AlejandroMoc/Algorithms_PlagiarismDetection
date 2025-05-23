import cv2
from optical_flow import calculate_optical_flow

def test_optical_flow():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Read first frame
    ret, frame_prev = cap.read()
    if not ret:
        print("Error reading video stream")
        return
    
    # Convert first frame to grayscale
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        
    while True:
        # Read next frame
        ret, frame_next = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
            
        # Calculate optical flow
        optical_flow, flow_visualization = calculate_optical_flow(gray_prev, gray_next)
        
        # Draw flow vectors on original frame
        magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])
        mask = magnitude > 1  # Filter out small movements
        
        # Draw arrows for significant motion
        step_size = 16  # Sample every 16 pixels
        for y_coord in range(0, optical_flow.shape[0], step_size):
            for x_coord in range(0, optical_flow.shape[1], step_size):
                if mask[y_coord, x_coord]:
                    flow_x, flow_y = optical_flow[y_coord, x_coord]
                    cv2.arrowedLine(frame_next, 
                                  (x_coord, y_coord),
                                  (int(x_coord + flow_x), int(y_coord + flow_y)),
                                  (0, 255, 0),
                                  2)
        
        # Display results
        cv2.imshow('Original with Flow', frame_next)
        cv2.imshow('Optical Flow Visualization', flow_visualization)
        
        # Update previous frame
        gray_prev = gray_next.copy()
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_optical_flow()