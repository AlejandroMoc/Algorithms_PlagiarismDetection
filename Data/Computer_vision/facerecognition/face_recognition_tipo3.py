import cv2
import numpy as np

def find_faces(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    detected_faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    return img, detected_faces

def recognize_faces(img, recognizer=None, known_faces=None):
    if recognizer is None or known_faces is None:
        return None
        
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = find_faces(img)[1]
    
    results = []
    for (x, y, w, h) in faces:
        face_roi = gray_img[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)
        results.append((label, confidence))
        
    return results

def train_face_recognizer(training_imgs, labels):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    training_data = []
    for img in training_imgs:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        training_data.append(gray_img)
    
    face_recognizer.train(training_data, np.array(labels))
    return face_recognizer

if __name__ == "__main__":
    test_img = cv2.imread('test_image.jpg')
    if test_img is None:
        print("Error: Could not load test image")
        exit()
        
    result_img, detected_faces = find_faces(test_img.copy())
    
    cv2.imshow('Detected Faces', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # To perform face recognition, load known faces and train the recognizer
    # known_faces = load_known_faces()  # Load known face database
    # face_recognizer = train_face_recognizer(known_faces['images'], known_faces['labels'])
    # recognition_results = recognize_faces(test_img, face_recognizer, known_faces)