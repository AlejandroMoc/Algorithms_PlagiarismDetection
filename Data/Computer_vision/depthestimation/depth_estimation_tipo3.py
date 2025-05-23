import cv2
import matplotlib.pyplot as plt

def estimacion_profundidad(imagen_izquierda, imagen_derecha):
    """
    Estima un mapa de profundidad a partir de imágenes estéreo usando Semi-Global Block Matching
    
    Args:
        imagen_izquierda: Imagen estéreo izquierda
        imagen_derecha: Imagen estéreo derecha
        
    Returns:
        Mapa de profundidad
    """
    # Convertir imágenes a escala de grises
    gris_izquierda = cv2.cvtColor(imagen_izquierda, cv2.COLOR_BGR2GRAY)
    gris_derecha = cv2.cvtColor(imagen_derecha, cv2.COLOR_BGR2GRAY)
    
    # Crear objeto matcher estéreo
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*16,
        blockSize=11,
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    # Calcular mapa de disparidad
    disparidad = stereo.compute(gris_izquierda, gris_derecha)
    
    # Normalizar mapa de disparidad para visualización
    disparidad_normalizada = cv2.normalize(disparidad, None, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return disparidad_normalizada

if __name__ == "__main__":
    # Cargar imágenes estéreo
    imagen_izquierda = cv2.imread('left_image.jpg')
    imagen_derecha = cv2.imread('right_image.jpg')
    
    if imagen_izquierda is None or imagen_derecha is None:
        print("Error: No se pudieron cargar las imágenes")
        exit()
        
    # Obtener mapa de profundidad
    mapa_profundidad = estimacion_profundidad(imagen_izquierda, imagen_derecha)
    
    # Mostrar resultados
    plt.figure(figsize=(12, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(imagen_izquierda, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Izquierda')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(imagen_derecha, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Derecha')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(mapa_profundidad, cmap='hot')
    plt.title('Mapa de Profundidad')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()