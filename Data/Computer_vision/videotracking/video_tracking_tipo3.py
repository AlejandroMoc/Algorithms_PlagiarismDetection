import cv2

def seguir_objeto_en_video(ruta_video):
    # Inicializar captura de video
    captura = cv2.VideoCapture(ruta_video)
    
    # Leer primer fotograma
    ret, fotograma = captura.read()
    if not ret:
        print("Fallo al leer video")
        return
    
    # Seleccionar ROI (Región de Interés)
    roi = cv2.selectROI(fotograma)
    
    # Inicializar tracker
    rastreador = cv2.TrackerCSRT_create()
    rastreador.init(fotograma, roi)
    
    while True:
        ret, fotograma = captura.read()
        if not ret:
            break
        
        # Actualizar tracker
        exito, caja = rastreador.update(fotograma)
        
        if exito:
            # Dibujar cuadro delimitador
            x, y, w, h = [int(v) for v in caja]
            cv2.rectangle(fotograma, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Mostrar resultado
        cv2.imshow('Seguimiento', fotograma)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ejemplo de uso
    archivo_video = "video_ejemplo.mp4"  # Reemplazar con tu archivo de video
    seguir_objeto_en_video(archivo_video)