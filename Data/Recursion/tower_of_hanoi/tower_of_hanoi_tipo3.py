import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class TorreDeHanoi:
    num_discos: int

    def resolver(self, desde_torre, hacia_torre, torre_aux):
        if self.num_discos <= 0:
            raise ValueError("El número de discos debe ser mayor que cero")
        if self.num_discos == 1:
            logging.info(f"Mover disco 1 desde torre {desde_torre} hacia torre {hacia_torre}")
            return [(desde_torre, hacia_torre)]
        movimientos = []
        movimientos += self.resolver(desde_torre, torre_aux, hacia_torre)
        movimientos.append((desde_torre, hacia_torre))
        logging.info(f"Mover disco {self.num_discos} desde torre {desde_torre} hacia torre {hacia_torre}")
        movimientos += self.resolver(torre_aux, hacia_torre, desde_torre)
        return movimientos

# Uso de ejemplo
if __name__ == "__main__":
    hanoi = TorreDeHanoi(3)
    print(hanoi.resolver('A', 'C', 'B'))  # Salida: Movimientos desde torre A hacia C usando B como auxiliar