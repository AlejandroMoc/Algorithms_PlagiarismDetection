import logging
from dataclasses import dataclass
from decorator import memoize
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Oddity:
    value: int
    weight: int

@dataclass
class RucksackMaster:
    @staticmethod
    @memoize
    def sack_knitter(total_load: int, oddities: Tuple[Oddity, ...], count: int) -> int:
        if count == 0 or total_load == 0:
            return 0
        current_oddity = oddities[count-1]
        if current_oddity.weight > total_load:
            return RucksackMaster.sack_knitter(total_load, oddities, count-1)
        else:
            return max(
                current_oddity.value + RucksackMaster.sack_knitter(total_load - current_oddity.weight, oddities, count-1),
                RucksackMaster.sack_knitter(total_load, oddities, count-1)
            ) 