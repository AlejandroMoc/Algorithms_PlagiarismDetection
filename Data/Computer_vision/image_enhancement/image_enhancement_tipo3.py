import cv2
import numpy as np

def mejorar_imagen(imagen):
    if imagen is None or len(imagen.shape) != 3:
        raise ValueError("Imagen de entrada inválida")

    yuv = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)

    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])

    yuv[:,:,0] = cv2.GaussianBlur(yuv[:,:,0], (3,3), 0)

    yuv[:,:,0] = cv2.convertScaleAbs(yuv[:,:,0], alpha=1.2, beta=5)

    imagen_mejorada = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    imagen_mejorada = cv2.filter2D(imagen_mejorada, -1, kernel)

    imagen_mejorada = np.clip(imagen_mejorada, 0, 255)
    imagen_mejorada = imagen_mejorada.astype(np.uint8)

    return imagen_mejorada