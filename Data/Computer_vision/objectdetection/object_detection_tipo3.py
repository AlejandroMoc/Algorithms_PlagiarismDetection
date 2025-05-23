import cv2
import numpy as np

def detect_objects(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained Haar Cascade classifier for face detection
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    detected_faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around detected faces
    processed_image = image.copy()
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(processed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    # Load YOLO model for general object detection
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layers = net.getLayerNames()
    output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Prepare image for YOLO
    img_height, img_width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Get detections
    outs = net.forward(output_layers)
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for output in outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                width = int(detection[2] * img_width)
                height = int(detection[3] * img_height)
                
                # Rectangle coordinates
                x_cord = int(center_x - width / 2)
                y_cord = int(center_y - height / 2)
                
                boxes.append([x_cord, y_cord, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw boxes for detected objects
    for idx in range(len(boxes)):
        if idx in indexes:
            x, y, w, h = boxes[idx]
            cv2.rectangle(processed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return processed_image