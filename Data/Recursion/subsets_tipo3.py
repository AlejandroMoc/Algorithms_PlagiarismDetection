import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class Subsets:
    conjunto: list

    def generar(self):
        return self._subconjuntos_recursivo([], sorted(self.conjunto))

    def _subconjuntos_recursivo(self, actual, conjunto):
        if not conjunto:
            logging.info(f"Subconjunto: {actual}")
            return [actual]
        return self._subconjuntos_recursivo(actual + [conjunto[0]], conjunto[1:]) + self._subconjuntos_recursivo(actual, conjunto[1:])

# Uso de ejemplo
if __name__ == "__main__":
    generador_subconjuntos = Subsets([1, 2, 3])
    print(generador_subconjuntos.generar())  # Salida: Todos los subconjuntos de [1, 2, 3] 