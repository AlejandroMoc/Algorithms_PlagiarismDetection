import logging
from dataclasses import dataclass
from decorator import memoize
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Element:
    value: int
    weight: int

@dataclass
class SolverKnapsack:
    @staticmethod
    @memoize
    def solve_knapsack(max_weight: int, elements: Tuple[Element, ...], index: int) -> int:
        if index == 0 or max_weight == 0:
            return 0
        current_element = elements[index-1]
        if current_element.weight > max_weight:
            return SolverKnapsack.solve_knapsack(max_weight, elements, index-1)
        else:
            return max(
                current_element.value + SolverKnapsack.solve_knapsack(max_weight - current_element.weight, elements, index-1),
                SolverKnapsack.solve_knapsack(max_weight, elements, index-1)
            )