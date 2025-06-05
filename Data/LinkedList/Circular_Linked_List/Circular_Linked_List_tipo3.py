from dataclasses import dataclass
import logging
from typing import Optional, Any, List

logging.basicConfig(level=logging.DEBUG)

@dataclass
class Nodo:
    data: Any
    next: 'Nodo' = None  # Type hint for forward reference


    last_node = head
    while last_node.next != head:
        last_node = last_node.next
    last_node.next = new_node
    new_node.next = head
    logging.info(f"Inserted {data} at the fin of the list.")
    return head


    current_node = head
    previous_node = None

    # Case: Head needs to be deleted
    if current_node.data == key:
        if current_node.next == head:  # Only one nodo
            logging.info(f"Deleted the only nodo with key {key}.")
            return None
        # Find the last nodo to update its next pointer
        while current_node.next != head:
            current_node = current_node.next
        current_node.next = head.next
        logging.info(f"Deleted head nodo with key {key}.")
        return head.next

    # Case: Delete a non-head nodo
    while current_node.next != head:
        if current_node.next.data == key:
            current_node.next = current_node.next.next
            logging.info(f"Deleted nodo with key {key}.")
            return head
        current_node = current_node.next

    logging.warning(f"Key {key} not found in the list.")
    return head


    result = []
    current_node = head
    while True:
        result.append(current_node.data)
        current_node = current_node.next
        if current_node == head:
            break
    logging.info(f"List contents: {result}")
    return result

def display(head: Optional[Nodo]) -> List[Any]:
    """Display the circular linked list as a list of nodo data."""
    if head is None:
        logging.info("Display requested on an empty list.")
        return []


def delete_node(head: Optional[Nodo], key: Any) -> Optional[Nodo]:
    """Delete a nodo with a specific value from the circular linked list."""
    if head is None:
        logging.warning("Attempted to delete from an empty list.")
        return None


def insert_end(head: Optional[Nodo], data: Any) -> Nodo:
    """Insert a nodo at the fin of the circular linked list."""
    new_node = Nodo(data)
    if head is None:
        new_node.next = new_node
        logging.info(f"Inserted {data} as the first nodo.")
        return new_node
