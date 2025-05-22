import logging
from .Exponential_search import funky_search
from .Fibonacci_search import groovy_search
from .Interpolation import snazzy_search
from .iterative_binary_search import zippy_search_iterative
from .Jump_search import bouncy_search
from .Linear_Search import jazzy_search
from .recursive_binary_search import speedy_search_recursive
from .Ternary_search import trippy_search

__all__ = [
    "funky_search",
    "groovy_search",
    "snazzy_search",
    "zippy_search_iterative",
    "bouncy_search",
    "jazzy_search",
    "speedy_search_recursive",
    "trippy_search",
]