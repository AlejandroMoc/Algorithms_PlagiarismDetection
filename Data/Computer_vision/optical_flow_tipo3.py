import cv2
import numpy as np

def calcular_flujo_optico(frame_anterior, frame_siguiente):
    # Convertir frames a escala de grises
    gris_anterior = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
    gris_siguiente = cv2.cvtColor(frame_siguiente, cv2.COLOR_BGR2GRAY)
    
    # Calcular flujo óptico usando el método Farneback
    flujo = cv2.calcOpticalFlowFarneback(gris_anterior, gris_siguiente, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Crear visualización
    magnitud, angulo = cv2.cartToPolar(flujo[..., 0], flujo[..., 1])
    
    # Crear imagen HSV para visualización
    hsv = np.zeros_like(frame_anterior)
    hsv[..., 1] = 255  # Saturación
    
    # Usar el ángulo para matiz y la magnitud para valor
    hsv[..., 0] = angulo * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convertir HSV a BGR para visualización
    visualizacion_flujo = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return flujo, visualizacion_flujo