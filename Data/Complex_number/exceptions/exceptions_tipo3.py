class NumeroComplejoError(Exception):
    """Clase base de excepción para operaciones con números complejos."""
    pass

class ErrorDivisionPorCero(NumeroComplejoError):
    """Excepción lanzada al intentar dividir por cero."""
    pass