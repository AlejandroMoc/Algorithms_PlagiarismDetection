from dataclasses import dataclass
import logging
from typing import Optional, Any, List

logging.basicConfig(level=logging.DEBUG)

@dataclass
class Spindle:
    info: Any
    next_node: 'Spindle' = None  # Type hint for forward reference

def add_end(leader: Optional[Spindle], info: Any) -> Spindle:
    """Incorporate a spindle at the end of the circular linked list."""
    new_spindle = Spindle(info)
    if leader is None:
        new_spindle.next_node = new_spindle
        logging.info(f"Inserted {info} as the first spindle.")
        return new_spindle

    final_spindle = leader
    while final_spindle.next_node != leader:
        final_spindle = final_spindle.next_node
    final_spindle.next_node = new_spindle
    new_spindle.next_node = leader
    logging.info(f"Inserted {info} at the end of the list.")
    return leader

def erase_spindle(leader: Optional[Spindle], target: Any) -> Optional[Spindle]:
    """Remove a spindle with a specific information from the circular linked list."""
    if leader is None:
        logging.warning("Attempted to delete from an empty list.")
        return None

    current_spindle = leader
    previous_spindle = None

    # Case: Leader needs to be deleted
    if current_spindle.info == target:
        if current_spindle.next_node == leader:  # Only one spindle
            logging.info(f"Deleted the only spindle with target {target}.")
            return None
        # Find the last spindle to update its next pointer
        while current_spindle.next_node != leader:
            current_spindle = current_spindle.next_node
        current_spindle.next_node = leader.next_node
        logging.info(f"Deleted leader spindle with target {target}.")
        return leader.next_node

    # Case: Delete a non-leader spindle
    while current_spindle.next_node != leader:
        if current_spindle.next_node.info == target:
            current_spindle.next_node = current_spindle.next_node.next_node
            logging.info(f"Deleted spindle with target {target}.")
            return leader
        current_spindle = current_spindle.next_node

    logging.warning(f"Target {target} not found in the list.")
    return leader

def showcase(leader: Optional[Spindle]) -> List[Any]:
    """Show the circular linked list as a list of spindle information."""
    if leader is None:
        logging.info("Display requested on an empty list.")
        return []

    result = []
    current_spindle = leader
    while True:
        result.append(current_spindle.info)
        current_spindle = current_spindle.next_node
        if current_spindle == leader:
            break
    logging.info(f"List contents: {result}")
    return result