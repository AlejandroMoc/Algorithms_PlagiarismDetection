import random
import string
from validators import PasswordValidator
from hasher import Hasher
from logging_config import logger

class ClaveGenerator:
    def __init__(self, longitud=12, usar_mayusculas=True, usar_minusculas=True, usar_digitos=True, usar_simbolos=True):
        self.longitud = longitud
        self.usar_mayusculas = usar_mayusculas
        self.usar_minusculas = usar_minusculas
        self.usar_digitos = usar_digitos
        self.usar_simbolos = usar_simbolos
        self.validador = PasswordValidator()
        self.hasher = Hasher()
        logger.info("ClaveGenerator inicializado con longitud=%d, mayusculas=%s, minusculas=%s, digitos=%s, simbolos=%s",
                    longitud, usar_mayusculas, usar_minusculas, usar_digitos, usar_simbolos)

    def generar_clave(self):
        try:
            pool_caracteres = ''
            caracteres_requeridos = []
            if self.usar_mayusculas:
                pool_caracteres += string.ascii_uppercase
                caracteres_requeridos.append(random.choice(string.ascii_uppercase))
            if self.usar_minusculas:
                pool_caracteres += string.ascii_lowercase
                caracteres_requeridos.append(random.choice(string.ascii_lowercase))
            if self.usar_digitos:
                pool_caracteres += string.digits
                caracteres_requeridos.append(random.choice(string.digits))
            if self.usar_simbolos:
                pool_caracteres += string.punctuation
                caracteres_requeridos.append(random.choice(string.punctuation))
            if not pool_caracteres:
                raise ValueError("Al menos un tipo de caracter debe ser seleccionado.")
            longitud_restante = self.longitud - len(caracteres_requeridos)
            if longitud_restante < 0:
                raise ValueError("La longitud de la clave es menor al número de tipos de caracteres requeridos.")
            clave = ''.join(random.choice(pool_caracteres) for _ in range(longitud_restante))
            clave = ''.join(random.sample(caracteres_requeridos + list(clave), self.longitud))
            if not self.validador.validate(clave):
                raise ValueError("La clave generada no cumple con los criterios de validación.")
            clave_hasheada = self.hasher.hash_password(clave)
            logger.info("Clave generada y hasheada exitosamente.")
            return clave, clave_hasheada
        except Exception as e:
            logger.error("Error generando clave: %s", str(e))
            raise 