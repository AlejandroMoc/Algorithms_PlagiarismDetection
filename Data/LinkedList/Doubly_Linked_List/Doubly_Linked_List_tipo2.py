from dataclasses import dataclass
from typing import Optional, Any, List
import logging

logging.basicConfig(level=logging.DEBUG)

@dataclass
class EleNode:
    data: Any
    next: Optional['EleNode'] = None
    prev: Optional['EleNode'] = None

def add_end(first: Optional[EleNode], data: Any) -> EleNode:
    """Add a node at the end of the doubly linked list."""
    new_ele = EleNode(data)
    if first is None:
        logging.info(f"Added {data} as the first element.")
        return new_ele

    last_ele = first
    while last_ele.next:
        last_ele = last_ele.next
    last_ele.next = new_ele
    new_ele.prev = last_ele
    logging.info(f"Added {data} at the end of the list.")
    return first

def remove_ele(first: Optional[EleNode], key: Any) -> Optional[EleNode]:
    """Remove a node with a specific value from the doubly linked list."""
    if first is None:
        logging.warning("Tried to remove from an empty list.")
        return None

    current_ele = first

    # Case: The first element contains the key to be removed
    if current_ele.data == key:
        first = current_ele.next
        if first:
            first.prev = None
        logging.info(f"Removed first element with key {key}.")
        return first

    # Case: Search for the key in the rest of the list
    while current_ele is not None and current_ele.data != key:
        current_ele = current_ele.next

    # Key not found
    if current_ele is None:
        logging.warning(f"Key {key} not found in the list.")
        return first

    # Adjust pointers to remove current_ele
    if current_ele.next:
        current_ele.next.prev = current_ele.prev
    if current_ele.prev:
        current_ele.prev.next = current_ele.next
    logging.info(f"Removed element with key {key}.")
    return first

def show(first: Optional[EleNode]) -> List[Any]:
    """Show the doubly linked list as a list of element data."""
    result = []
    current_ele = first
    while current_ele:
        result.append(current_ele.data)
        current_ele = current_ele.next
    logging.info(f"List contents: {result}")
    return result