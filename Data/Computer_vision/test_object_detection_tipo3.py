import cv2
import numpy as np
from object_detection import detect_objects

# Crear una imagen con formas aleatorias
imagen_prueba = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.rectangle(imagen_prueba, (20, 20), (80, 80), (0, 255, 0), -1)  # Objeto simulado

# Detectar objetos en la imagen
objetos_detectados = detect_objects(imagen_prueba)

# Dibujar rectángulos delimitadores alrededor de los objetos detectados
imagen_salida = imagen_prueba.copy()
if objetos_detectados:
    for (x, y, w, h) in objetos_detectados:
        # Dibujar rectángulo alrededor del objeto detectado
        cv2.rectangle(imagen_salida, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Agregar etiqueta
        cv2.putText(imagen_salida, 'Objeto', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Mostrar la imagen original y los objetos detectados
cv2.imshow('Imagen Original', imagen_prueba)
cv2.imshow('Detección de Objetos', imagen_salida)
cv2.waitKey(0)
cv2.destroyAllWindows()