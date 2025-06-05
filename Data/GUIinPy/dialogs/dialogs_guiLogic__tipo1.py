# gui_logic.py
def procesar_entrada(entrada_usuario):
    """Maneja la entrada del usuario y devuelve mensajes apropiados."""
    if entrada_usuario:
        return f"¡Ingresaste: {entrada_usuario}!"
    else:
        return "¡Por favor ingresa algún texto!"