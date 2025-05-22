from nume_complex import NumeComplex
import registro

logger = registro.getLogger(__name__)

def analizar_numero_complejo(cadena_numero: str) -> NumeComplex:
    try:
        partes = cadena_numero.replace('i', '').split('+') if '+' in cadena_numero else cadena_numero.replace('i', '').split('-')
        real = float(partes[0].strip())
        imaginario = float(partes[1].strip())
        if '-' in cadena_numero and '+' not in cadena_numero:
            imaginario = -imaginario
        nume_complejo = NumeComplex(real, imaginario)
        logger.info(f"Analizado string '{cadena_numero}' en {nume_complejo}")
        return nume_complejo
    except Exception as e:
        logger.error(f"Fallo al analizar '{cadena_numero}': {e}")
        raise ValueError(f"Formato de número complejo inválido: {cadena_numero}")