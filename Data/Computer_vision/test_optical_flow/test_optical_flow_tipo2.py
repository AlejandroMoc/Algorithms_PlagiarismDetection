import cv2
from optical_flow import calculate_optical_flow

def funky_flow_analysis():
    # Initialize video capture
    vid_cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not vid_cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Read first frame
    ret, initial_frame = vid_cap.read()
    if not ret:
        print("Error reading video stream")
        return
    
    # Convert first frame to grayscale
    initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        
    while True:
        # Read next frame
        ret, next_frame = vid_cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            
        # Calculate optical flow
        flow_data, vis_flow = calculate_optical_flow(initial_gray, next_gray)
        
        # Draw flow vectors on original frame
        magnitude, angle = cv2.cartToPolar(flow_data[..., 0], flow_data[..., 1])
        mask = magnitude > 1  # Filter out small movements
        
        # Draw arrows for significant motion
        step = 16  # Sample every 16 pixels
        for y in range(0, flow_data.shape[0], step):
            for x in range(0, flow_data.shape[1], step):
                if mask[y, x]:
                    dx, dy = flow_data[y, x]
                    cv2.arrowedLine(next_frame, 
                                  (x, y),
                                  (int(x + dx), int(y + dy)),
                                  (0, 255, 0),
                                  2)
        
        # Display results
        cv2.imshow('Original with Flow', next_frame)
        cv2.imshow('Optical Flow Visualization', vis_flow)
        
        # Update previous frame
        initial_gray = next_gray.copy()
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    vid_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    funky_flow_analysis()