import cv2
import numpy as np
import os

def detect_faces(img, cascade_file):
    # Check if cascade file exists
    if not os.path.exists(cascade_file):
        raise FileNotFoundError(f"Cascade file not found at {cascade_file}")
    
    # Create a copy of the image to draw on
    output_img = img.copy()
    
    # Convert to grayscale for face detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cascade_file)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return output_img

# Create test image with a simple face-like pattern
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(test_img, (30, 30), (70, 70), (255, 255, 255), -1)  # Face
cv2.rectangle(test_img, (45, 40), (55, 50), (0, 0, 0), -1)  # Eyes
cv2.rectangle(test_img, (40, 60), (60, 65), (0, 0, 0), -1)  # Mouth

# Process the image with face recognition
cascade_file = 'haarcascade_frontalface_default.xml'
recognized_img = detect_faces(test_img, cascade_file)

# Display the image
cv2.imshow('Face Recognition', recognized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()