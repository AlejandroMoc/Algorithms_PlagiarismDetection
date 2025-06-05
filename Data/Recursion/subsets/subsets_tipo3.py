import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class Subsets:
    set: list

    
    
# Sample usage
if __name__ == "__main__":
    subset_gen = Subsets([1, 2, 3])
    print(subset_gen.generate())  # Output: All subsets of [1, 2, 3] 

def _subsets_recursive(self, current, set):
        if not set:
            logging.info(f"Subset: {current}")
            return [current]
        return self._subsets_recursive(current + [set[0]], set[1:]) + self._subsets_recursive(current, set[1:])


def generate(self):
        return self._subsets_recursive([], sorted(self.set))
