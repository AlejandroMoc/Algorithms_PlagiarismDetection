import logging
from dataclasses import dataclass
from decorator import memoize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SolvingMachine:
    @staticmethod
    @memoize
    def magic_spell(A: str, B: str, x: int, y: int) -> int:
        if x == 0 or y == 0:
            return 0
        if A[x-1] == B[y-1]:
            return 1 + SolvingMachine.magic_spell(A, B, x-1, y-1)
        else:
            return max(SolvingMachine.magic_spell(A, B, x, y-1), SolvingMachine.magic_spell(A, B, x-1, y)) 