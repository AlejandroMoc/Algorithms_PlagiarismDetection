import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class BusquedaBinaria:
    lista: list
    objetivo: int

    def buscar(self) -> int:
        return self._busqueda_binaria_recursiva(self.lista, self.objetivo, 0, len(self.lista) - 1)

    def _busqueda_binaria_recursiva(self, lista, objetivo, izquierda, derecha):
        if derecha >= izquierda:
            medio = izquierda + (derecha - izquierda) // 2
            if lista[medio] == objetivo:
                logging.info(f"Objetivo {objetivo} encontrado en el índice {medio}")
                return medio
            elif lista[medio] > objetivo:
                return self._busqueda_binaria_recursiva(lista, objetivo, izquierda, medio - 1)
            else:
                return self._busqueda_binaria_recursiva(lista, objetivo, medio + 1, derecha)
        else:
            logging.info(f"Objetivo {objetivo} no encontrado")
            return -1

# Uso de ejemplo
if __name__ == "__main__":
    bb_instancia = BusquedaBinaria([1, 2, 3, 4, 5], 3)
    print(f"Índice de 3 es {bb_instancia.buscar()}")  # Salida: 2 