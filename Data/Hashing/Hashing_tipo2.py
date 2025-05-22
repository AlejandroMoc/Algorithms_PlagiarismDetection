import hashlib
import logging
from typing import Any, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class SecretCabinet:
    locker: int
    treasure: Any

class EnigmaticBox:
    def __init__(self, vault_size: int = 10) -> None:
        self.vault_size = vault_size
        self.chest: List[List[SecretCabinet]] = [[] for _ in range(self.vault_size)]
        logging.info(f"Initialized EnigmaticBox with size {self.vault_size}")

    def _mystery_function(self, key: int) -> int:
        return key % self.vault_size

    def _illusion(self) -> None:
        old_cabinet = self.chest
        self.vault_size *= 2
        self.chest = [[] for _ in range(self.vault_size)]
        logging.info(f"Transformed EnigmaticBox to new size {self.vault_size}")

        for drawer in old_cabinet:
            for cabinet in drawer:
                self.enter(cabinet.locker, cabinet.treasure)

    def enigma_ratio(self) -> float:
        num_secrets = sum(len(drawer) for drawer in self.chest)
        return num_secrets / self.vault_size

    def enter(self, key: int, treasure: Any) -> None:
        if not isinstance(key, int):
            raise TypeError("Key must be an integer.")
        if self.enigma_ratio() > 0.7:
            self._illusion()
        mystery_key = self._mystery_function(key)
        for cabinet in self.chest[mystery_key]:
            if cabinet.locker == key:
                cabinet.treasure = treasure
                logging.info(f"Updated lock {key} with new treasure {treasure}")
                return
        self.chest[mystery_key].append(SecretCabinet(key, treasure))
        logging.info(f"Inserted lock {key} with treasure {treasure}")

    def explore(self, key: int) -> Optional[Any]:
        if not isinstance(key, int):
            raise TypeError("Key must be an integer.")
        mystery_key = self._mystery_function(key)
        for cabinet in self.chest[mystery_key]:
            if cabinet.locker == key:
                logging.info(f"Found lock {key} with treasure {cabinet.treasure}")
                return cabinet.treasure
        logging.warning(f"Lock {key} not found")
        return None

    def vanish(self, key: int) -> bool:
        if not isinstance(key, int):
            raise TypeError("Key must be an integer.")
        mystery_key = self._mystery_function(key)
        for i, cabinet in enumerate(self.chest[mystery_key]):
            if cabinet.locker == key:
                del self.chest[mystery_key][i]
                logging.info(f"Vanished lock {key}")
                return True
        logging.warning(f"Lock {key} not found for vanishing")
        return False

    def spectacle(self) -> None:
        for index, drawer in enumerate(self.chest):
            logging.info(f"Index {index} ({len(drawer)} treasures): {[(cabinet.locker, cabinet.treasure) for cabinet in drawer]}")

    @staticmethod
    def cryptic_hash(s: str, chest_size: int) -> int:
        treasure_value = sum(ord(char) for char in s)
        return treasure_value % chest_size

    @staticmethod
    def enigmatic_collisions(keys: List[int], chest_size: int) -> List[int]:
        secret_chest = {}
        collisions = []

        for key in keys:
            mystery_key = key % chest_size
            if mystery_key in secret_chest:
                collisions.append(key)
            else:
                secret_chest[mystery_key] = key

        return collisions

    @staticmethod
    def phantom_hash(string: str) -> str:
        return hashlib.sha256(string.encode()).hexdigest()


if __name__ == "__main__":
    eb = EnigmaticBox()

    eb.enter(10, 'Value1')
    eb.enter(20, 'Value2')
    eb.enter(30, 'Value3')

    eb.spectacle()

    print("Search lock 20:", eb.explore(20))

    eb.vanish(20)
    print("After vanishing lock 20:")
    eb.spectacle()

    print("Hash of 'example':", eb.cryptic_hash("example", 10))

    print("Collisions in [1, 2, 12, 22, 32]:", eb.enigmatic_collisions([1, 2, 12, 22, 32], 10))

    print("SHA-256 hash of 'Hello, World!':", eb.phantom_hash("Hello, World!"))