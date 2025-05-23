from PIL import Image, ImageFilter

def cambiar_tamaño(ruta_imagen, ruta_salida, tamaño):
    """
    Cambia el tamaño de la imagen al tamaño especificado.
    """
    with Image.open(ruta_imagen) as img:
        img = img.resize(tamaño, Image.ANTIALIAS)  # Utiliza ANTIALIAS para mejor calidad
        img.save(ruta_salida)

def aplicar_filtro(ruta_imagen, ruta_salida, tipo_filtro):
    """
    Aplica un filtro a la imagen y la guarda.
    """
    with Image.open(ruta_imagen) as img:
        if tipo_filtro == 'DESENFOQUE':
            img = img.filter(ImageFilter.BLUR)
        elif tipo_filtro == 'CONTORNO':
            img = img.filter(ImageFilter.CONTOUR)
        img.save(ruta_salida)

def rotar_imagen(ruta_imagen, ruta_salida, ángulo):
    """
    Rota la imagen por el ángulo especificado.
    """
    with Image.open(ruta_imagen) as img:
        img = img.rotate(ángulo, expand=True)  # Expandir para ajustar al nuevo tamaño
        img.save(ruta_salida)