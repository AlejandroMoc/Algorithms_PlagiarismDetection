import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class Permutations:
    cadena: str

    def generar(self):
        return self._permutar(self.cadena)

    def _permutar(self, s, respuesta=''):
        if len(s) == 0:
            logging.info(f"Permutación: {respuesta}")
            return [respuesta]
        resultados = []
        i = 0
        while i < len(s):
            caracter = s[i]
            subcadena_izquierda = s[0:i]
            subcadena_derecha = s[i + 1:]
            resultados += self._permutar(subcadena_izquierda + subcadena_derecha, respuesta + caracter)
            i += 1
        return resultados

# Uso de ejemplo
if __name__ == "__main__":
    perm = Permutations("abc")
    print(perm.generar())  # Salida: Todas las permutaciones de 'abc'