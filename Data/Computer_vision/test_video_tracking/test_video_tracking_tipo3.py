import cv2

def rastrear_objeto_en_video(ruta_video):
    cap = cv2.VideoCapture(ruta_video)
    
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el video")
        return
        
    roi = cv2.selectROI("Seleccionar Objeto", frame, False)
    cv2.destroyWindow("Seleccionar Objeto")
    
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, roi)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        success, box = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        cv2.imshow("Rastreo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ruta_video = "video_ejemplo.mp4"
    rastrear_objeto_en_video(ruta_video)