python
from dataclasses import dataclass
import logging
from typing import Optional, Any, List

logging.basicConfig(level=logging.DEBUG)

@dataclass
class Nodo:
    dato: Any
    siguiente: 'Nodo' = None  # Type hint for forward reference

def insertar_final(cabeza: Optional[Nodo], dato: Any) -> Nodo:
    """Insert a node at the end of the circular linked list."""
    nuevo_nodo = Nodo(dato)
    if cabeza is None:
        nuevo_nodo.siguiente = nuevo_nodo
        logging.info(f"Inserted {dato} as the first node.")
        return nuevo_nodo

    ultimo_nodo = cabeza
    while ultimo_nodo.siguiente != cabeza:
        ultimo_nodo = ultimo_nodo.siguiente
    ultimo_nodo.siguiente = nuevo_nodo
    nuevo_nodo.siguiente = cabeza
    logging.info(f"Inserted {dato} at the end of the list.")
    return cabeza

def eliminar_nodo(cabeza: Optional[Nodo], clave: Any) -> Optional[Nodo]:
    """Delete a node with a specific value from the circular linked list."""
    if cabeza is None:
        logging.warning("Attempted to delete from an empty list.")
        return None

    nodo_actual = cabeza
    nodo_anterior = None

    # Case: Head needs to be deleted
    if nodo_actual.dato == clave:
        if nodo_actual.siguiente == cabeza:  # Only one node
            logging.info(f"Deleted the only node with key {clave}.")
            return None
        # Find the last node to update its next pointer
        while nodo_actual.siguiente != cabeza:
            nodo_actual = nodo_actual.siguiente
        nodo_actual.siguiente = cabeza.siguiente
        logging.info(f"Deleted head node with key {clave}.")
        return cabeza.siguiente

    # Case: Delete a non-head node
    while nodo_actual.siguiente != cabeza:
        if nodo_actual.siguiente.dato == clave:
            nodo_actual.siguiente = nodo_actual.siguiente.siguiente
            logging.info(f"Deleted node with key {clave}.")
            return cabeza
        nodo_actual = nodo_actual.siguiente

    logging.warning(f"Key {clave} not found in the list.")
    return cabeza

def mostrar(cabeza: Optional[Nodo]) -> List[Any]:
    """Display the circular linked list as a list of node data."""
    if cabeza is None:
        logging.info("Display requested on an empty list.")
        return []

    resultado = []
    nodo_actual = cabeza
    while True:
        resultado.append(nodo_actual.dato)
        nodo_actual = nodo_actual.siguiente
        if nodo_actual == cabeza:
            break
    logging.info(f"List contents: {resultado}")
    return resultado