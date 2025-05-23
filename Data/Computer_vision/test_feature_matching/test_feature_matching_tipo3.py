import cv2

def buscar_caracteristicas(imagen1_ruta, imagen2_ruta):
    # Leer imágenes
    imagen1 = cv2.imread(imagen1_ruta)
    imagen2 = cv2.imread(imagen2_ruta)
    
    # Convertir a escala de grises
    gris1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    gris2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)

    # Inicializar detector SIFT
    sift = cv2.SIFT_create()

    # Encontrar keypoints y descriptores
    kp1, des1 = sift.detectAndCompute(gris1, None)
    kp2, des2 = sift.detectAndCompute(gris2, None)

    # Parámetros FLANN
    FLANN_INDEX_KDTREE = 1
    params_indice = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    params_busqueda = dict(checks=50)

    # Matcher FLANN
    flann = cv2.FlannBasedMatcher(params_indice, params_busqueda)
    coincidencias = flann.knnMatch(des1, des2, k=2)

    # Aplicar test de razón
    buenas_coincidencias = []
    for match1, match2 in coincidencias:
        if match1.distance < 0.7 * match2.distance:
            buenas_coincidencias.append(match1)

    # Dibujar coincidencias
    imagen_coincidencias = cv2.drawMatches(imagen1, kp1, imagen2, kp2, buenas_coincidencias, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return imagen_coincidencias

if __name__ == "__main__":
    # Ejemplo de uso
    imagen1_ruta = "imagen1.jpg"
    imagen2_ruta = "imagen2.jpg"
    
    resultado = buscar_caracteristicas(imagen1_ruta, imagen2_ruta)
    
    # Mostrar resultado
    cv2.imshow("Coincidencia de Características", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()