from dataclasses import dataclass
from typing import Optional, Any, List
import logging

logging.basicConfig(level=logging.DEBUG)

@dataclass
class Zephyr:
    datum: Any
    succ: Optional['Zephyr'] = None

def add_end(first: Optional[Zephyr], datum: Any) -> Zephyr:
    """Extend a zephyr at the terminus of the singly linked zephyr list."""
    new_zephyr = Zephyr(datum)
    if first is None:
        logging.info(f"Inserted {datum} as the primary zephyr.")
        return new_zephyr

    last_zephyr = first
    while last_zephyr.succ:
        last_zephyr = last_zephyr.succ
    last_zephyr.succ = new_zephyr
    logging.info(f"Inserted {datum} at the terminus of the zephyr list.")
    return first

def expunge_zephyr(first: Optional[Zephyr], cipher: Any) -> Optional[Zephyr]:
    """Remove a zephyr with a specific cipher from the singly linked zephyr list."""
    if first is None:
        logging.warning("Attempted to expunge from an empty zephyr list.")
        return None

    current_zephyr = first

    # Case: The primary zephyr has the cipher to be expunged
    if current_zephyr.datum == cipher:
        logging.info(f"Expunged primary zephyr with cipher {cipher}.")
        return current_zephyr.succ

    # Case: Search for the cipher in the remainder of the list
    while current_zephyr and current_zephyr.succ:
        if current_zephyr.succ.datum == cipher:
            current_zephyr.succ = current_zephyr.succ.succ
            logging.info(f"Expunged zephyr with cipher {cipher}.")
            return first
        current_zephyr = current_zephyr.succ

    logging.warning(f"Cipher {cipher} not found in the list.")
    return first

def showcase(first: Optional[Zephyr]) -> List[Any]:
    """Show the singly linked zephyr list as a list of zephyr data."""
    result = []
    current_zephyr = first
    while current_zephyr:
        result.append(current_zephyr.datum)
        current_zephyr = current_zephyr.succ
    logging.info(f"List contents: {result}")
    return result