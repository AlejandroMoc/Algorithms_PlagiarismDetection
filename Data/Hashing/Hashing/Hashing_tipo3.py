import hashlib
import logging
from typing import Any, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Entry:
    key: int
    value: Any

class HashMap:
    def __init__(self, capacity: int = 10) -> None:
        self.capacity = capacity
        self.map: List[List[Entry]] = [[] for _ in range(self.capacity)]
        logging.info(f"Initialized HashMap with capacity {self.capacity}")

    def _hash(self, key: int) -> int:
        return key % self.capacity

    def _resize_map(self) -> None:
        old_map = self.map
        self.capacity *= 2
        self.map = [[] for _ in range(self.capacity)]
        logging.info(f"Resized HashMap to new capacity {self.capacity}")

        for bucket in old_map:
            for entry in bucket:
                self.add(entry.key, entry.value)

    def load_factor(self) -> float:
        num_elements = sum(len(bucket) for bucket in self.map)
        return num_elements / self.capacity

    def add(self, key: int, value: Any) -> None:
        if not isinstance(key, int):
            raise TypeError("Key must be an integer.")
        if self.load_factor() > 0.7:
            self._resize_map()
        hash_key = self._hash(key)
        for entry in self.map[hash_key]:
            if entry.key == key:
                entry.value = value
                logging.info(f"Updated key {key} with new value {value}")
                return
        self.map[hash_key].append(Entry(key, value))
        logging.info(f"Added key {key} with value {value}")

    def find(self, key: int) -> Optional[Any]:
        if not isinstance(key, int):
            raise TypeError("Key must be an integer.")
        hash_key = self._hash(key)
        for entry in self.map[hash_key]:
            if entry.key == key:
                logging.info(f"Found key {key} with value {entry.value}")
                return entry.value
        logging.warning(f"Key {key} not found")
        return None

    def remove(self, key: int) -> bool:
        if not isinstance(key, int):
            raise TypeError("Key must be an integer.")
        hash_key = self._hash(key)
        for i, entry in enumerate(self.map[hash_key]):
            if entry.key == key:
                del self.map[hash_key][i]
                logging.info(f"Removed key {key}")
                return True
        logging.warning(f"Key {key} not found for removal")
        return False

    def show_map(self) -> None:
        for index, bucket in enumerate(self.map):
            logging.info(f"Index {index} ({len(bucket)} entries): {[(entry.key, entry.value) for entry in bucket]}")

    @staticmethod
    def string_hash(s: str, capacity: int) -> int:
        hash_value = sum(ord(char) for char in s)
        return hash_value % capacity

    @staticmethod
    def find_collisions(keys: List[int], capacity: int) -> List[int]:
        hash_map = {}
        collisions = []

        for key in keys:
            hash_key = key % capacity
            if hash_key in hash_map:
                collisions.append(key)
            else:
                hash_map[hash_key] = key

        return collisions

    @staticmethod
    def sha256_hash(string: str) -> str:
        return hashlib.sha256(string.encode()).hexdigest()


if __name__ == "__main__":
    hm = HashMap()

    hm.add(10, 'Value1')
    hm.add(20, 'Value2')
    hm.add(30, 'Value3')

    hm.show_map()

    print("Find key 20:", hm.find(20))

    hm.remove(20)
    print("After removing key 20:")
    hm.show_map()

    print("Hash of 'example':", hm.string_hash("example", 10))

    print("Collisions in [1, 2, 12, 22, 32]:", hm.find_collisions([1, 2, 12, 22, 32], 10))

    print("SHA-256 hash of 'Hello, World!':", hm.sha256_hash("Hello, World!"))