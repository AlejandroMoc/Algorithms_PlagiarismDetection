import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)

@dataclass
class SpinningDisks:
    num_spins: int

    def spin(self, source, target, helper):
        if self.num_spins <= 0:
            raise ValueError("Number of spins must be greater than zero")
        if self.num_spins == 1:
            logging.info(f"Spin disk 1 from source {source} to target {target}")
            return [(source, target)]
        spins = []
        spins += self.spin(source, helper, target)
        spins.append((source, target))
        logging.info(f"Spin disk {self.num_spins} from source {source} to target {target}")
        spins += self.spin(helper, target, source)
        return spins

# Sample usage
if __name__ == "__main__":
    disks = SpinningDisks(3)
    print(disks.spin('A', 'C', 'B'))  # Output: Moves from source A to C using B as helper