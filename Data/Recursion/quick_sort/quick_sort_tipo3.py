import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class OrdenamientoRapido:
    lista: list

    def ordenar(self) -> list:
        return self._ordenamiento_rapido_recursivo(self.lista, 0, len(self.lista) - 1)

    def _ordenamiento_rapido_recursivo(self, lista, bajo, alto):
        if bajo < alto:
            pivote = self._particionar(lista, bajo, alto)
            self._ordenamiento_rapido_recursivo(lista, bajo, pivote - 1)
            self._ordenamiento_rapido_recursivo(lista, pivote + 1, alto)
        return lista

    def _particionar(self, lista, bajo, alto):
        pivote = lista[alto]
        i = bajo - 1
        for j in range(bajo, alto):
            if lista[j] <= pivote:
                i += 1
                lista[i], lista[j] = lista[j], lista[i]
        lista[i + 1], lista[alto] = lista[alto], lista[i + 1]
        return i + 1

# Uso de ejemplo
if __name__ == "__main__":
    or_instance = OrdenamientoRapido([10, 80, 30, 90, 40, 50, 70])
    print(f"Lista ordenada: {or_instance.ordenar()}")  # Salida: [10, 30, 40, 50, 70, 80, 90]