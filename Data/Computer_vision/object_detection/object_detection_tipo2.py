import cv2
import numpy as np

def find_objects(img):
    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained Haar Cascade classifier for face detection
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_detector.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around detected faces
    detected_img = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(detected_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Load YOLO model for general object detection
    yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    
    # Prepare image for YOLO
    h, w, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    
    # Get detections
    outs = yolo_net.forward(output_layers)
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                
                # Rectangle coordinates
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw boxes for detected objects
    for i in range(len(boxes)):
        if i in indexes:
            x, y, width, height = boxes[i]
            cv2.rectangle(detected_img, (x, y), (x + width, y + height), (255, 0, 0), 2)
    
    return detected_img