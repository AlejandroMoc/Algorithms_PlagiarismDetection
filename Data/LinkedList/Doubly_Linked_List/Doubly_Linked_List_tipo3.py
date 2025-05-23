from dataclasses import dataclass
from typing import Optional, Any, List
import logging

logging.basicConfig(level=logging.DEBUG)

@dataclass
class Nodo:
    dato: Any
    siguiente: Optional['Nodo'] = None
    anterior: Optional['Nodo'] = None

def insertar_final(cabeza: Optional[Nodo], dato: Any) -> Nodo:
    """Inserta un nodo al final de la lista doblemente enlazada."""
    nuevo_nodo = Nodo(dato)
    if cabeza is None:
        logging.info(f"Insertado {dato} como el primer nodo.")
        return nuevo_nodo

    ultimo_nodo = cabeza
    while ultimo_nodo.siguiente:
        ultimo_nodo = ultimo_nodo.siguiente
    ultimo_nodo.siguiente = nuevo_nodo
    nuevo_nodo.anterior = ultimo_nodo
    logging.info(f"Insertado {dato} al final de la lista.")
    return cabeza

def eliminar_nodo(cabeza: Optional[Nodo], clave: Any) -> Optional[Nodo]:
    """Elimina un nodo con un valor específico de la lista doblemente enlazada."""
    if cabeza is None:
        logging.warning("Intento de eliminar de una lista vacía.")
        return None

    nodo_actual = cabeza

    # Caso: El nodo de la cabeza tiene la clave que se va a eliminar
    if nodo_actual.dato == clave:
        cabeza = nodo_actual.siguiente
        if cabeza:
            cabeza.anterior = None
        logging.info(f"Nodo de la cabeza con clave {clave} eliminado.")
        return cabeza

    # Caso: Buscar la clave en el resto de la lista
    while nodo_actual is not None and nodo_actual.dato != clave:
        nodo_actual = nodo_actual.siguiente

    # Clave no encontrada
    if nodo_actual is None:
        logging.warning(f"Clave {clave} no encontrada en la lista.")
        return cabeza

    # Ajustar punteros para quitar nodo_actual
    if nodo_actual.siguiente:
        nodo_actual.siguiente.anterior = nodo_actual.anterior
    if nodo_actual.anterior:
        nodo_actual.anterior.siguiente = nodo_actual.siguiente
    logging.info(f"Nodo con clave {clave} eliminado.")
    return cabeza

def mostrar(cabeza: Optional[Nodo]) -> List[Any]:
    """Muestra la lista doblemente enlazada como una lista de datos de nodo."""
    resultado = []
    nodo_actual = cabeza
    while nodo_actual:
        resultado.append(nodo_actual.dato)
        nodo_actual = nodo_actual.siguiente
    logging.info(f"Contenido de la lista: {resultado}")
    return resultado