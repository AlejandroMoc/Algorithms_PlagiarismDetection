import cv2
import numpy as np
import os

def detect_faces(image, cascade_file_path):
    if not os.path.exists(cascade_file_path):
        raise FileNotFoundError(f"Cascade file not found at {cascade_file_path}")
    
    output_image = image.copy()
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cascade_file_path)
    
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return output_image

test_image = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(test_image, (30, 30), (70, 70), (255, 255, 255), -1)  
cv2.rectangle(test_image, (45, 40), (55, 50), (0, 0, 0), -1)  
cv2.rectangle(test_image, (40, 60), (60, 65), (0, 0, 0), -1)  

cascade_file = 'haarcascade_frontalface_default.xml'
recognized_image = detect_faces(test_image, cascade_file)

cv2.imshow('Face Recognition', recognized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()