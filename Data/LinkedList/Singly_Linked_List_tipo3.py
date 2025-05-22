python
from dataclasses import dataclass
from typing import Optional, Any, List
import logging

logging.basicConfig(level=logging.DEBUG)

@dataclass
class Nodo:
    dato: Any
    siguiente: Optional['Nodo'] = None

def insertar_final(cabeza: Optional[Nodo], dato: Any) -> Nodo:
    """Inserta un nodo al final de la lista enlazada."""
    nuevo_nodo = Nodo(dato)
    if cabeza is None:
        logging.info(f"Insertado {dato} como primer nodo.")
        return nuevo_nodo

    ultimo_nodo = cabeza
    while ultimo_nodo.siguiente:
        ultimo_nodo = ultimo_nodo.siguiente
    ultimo_nodo.siguiente = nuevo_nodo
    logging.info(f"Insertado {dato} al final de la lista.")
    return cabeza

def eliminar_nodo(cabeza: Optional[Nodo], clave: Any) -> Optional[Nodo]:
    """Elimina un nodo con un valor específico de la lista enlazada."""
    if cabeza is None:
        logging.warning("Intento de eliminar de una lista vacía.")
        return None

    nodo_actual = cabeza

    # Caso: El nodo cabeza tiene la clave a eliminar
    if nodo_actual.dato == clave:
        logging.info(f"Nodo cabeza con clave {clave} eliminado.")
        return nodo_actual.siguiente

    # Caso: Buscar la clave en el resto de la lista
    while nodo_actual and nodo_actual.siguiente:
        if nodo_actual.siguiente.dato == clave:
            nodo_actual.siguiente = nodo_actual.siguiente.siguiente
            logging.info(f"Nodo con clave {clave} eliminado.")
            return cabeza
        nodo_actual = nodo_actual.siguiente

    logging.warning(f"Clave {clave} no encontrada en la lista.")
    return cabeza

def mostrar(cabeza: Optional[Nodo]) -> List[Any]:
    """Muestra la lista enlazada como una lista de datos de nodo."""
    resultado = []
    nodo_actual = cabeza
    while nodo_actual:
        resultado.append(nodo_actual.dato)
        nodo_actual = nodo_actual.siguiente
    logging.info(f"Contenidos de la lista: {resultado}")
    return resultado