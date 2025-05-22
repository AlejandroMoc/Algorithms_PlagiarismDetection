class ErrorCorreoElectronico(Exception):
    """Excepción base para operaciones de correo electrónico."""
    pass

class ErrorConexionSMTP(ErrorCorreoElectronico):
    """Se levanta cuando falla la conexión SMTP."""
    pass

class ErrorAdjunto(ErrorCorreoElectronico):
    """Se levanta cuando falla el manejo de adjuntos."""
    pass

class ErrorPlantilla(ErrorCorreoElectronico):
    """Se levanta cuando falla el procesamiento de plantillas."""
    pass