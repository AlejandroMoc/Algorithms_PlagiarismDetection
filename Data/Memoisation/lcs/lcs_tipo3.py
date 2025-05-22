import logging
from dataclasses import dataclass
from decorator import memoize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LongestCommonSubsequenceSolver:
    @staticmethod
    @memoize
    def longest_common_subsequence(X: str, Y: str, m: int, n: int) -> int:
        if m == 0 or n == 0:
            return 0
        if X[m-1] == Y[n-1]:
            return 1 + LongestCommonSubsequenceSolver.longest_common_subsequence(X, Y, m-1, n-1)
        else:
            return max(LongestCommonSubsequenceSolver.longest_common_subsequence(X, Y, m, n-1), LongestCommonSubsequenceSolver.longest_common_subsequence(X, Y, m-1, n))