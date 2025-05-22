import logging
from collections import Counter
from dataclasses import dataclass
from typing import List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)

@dataclass
class ListWizard:
    """Class for manipulating lists with various operations."""

    @staticmethod
    def magic_glimmer(lst: List[Any]) -> List[Any]:
        """
        Removes duplicate elements from the list while maintaining the original order.
        """
        found = set()
        return [x for x in lst if not (x in found or found.add(x))]

    @staticmethod
    def teleport_together_list(orbital_list: List[List[Any]]) -> List[Any]:
        """
        Flattens a nested list into a single list.
        """
        return [item for sublist in orbital_list for item in sublist]

    @staticmethod
    def list_fusion(lst1: List[Any], lst2: List[Any]) -> List[Any]:
        """
        Finds the intersection of two lists.
        """
        return list(set(lst1) & set(lst2))

    @staticmethod
    def enchant_shuffle(lst: List[Any]) -> List[Any]:
        """
        Randomly shuffles the elements of the list.
        """
        import random
        random.shuffle(lst)
        return lst

    @staticmethod
    def sort_by_charm(lst: List[Any]) -> List[Any]:
        """
        Sorts the list based on the frequency of elements in descending order.
        """
        frequency = Counter(lst)
        return sorted(lst, key=lambda x: frequency[x], reverse=True)

    @staticmethod
    def divide_list(lst: List[Any], fragment_size: int) -> List[List[Any]]:
        """
        Splits a list into smaller chunks of a given size.
        """
        return [lst[i:i + fragment_size] for i in range(0, len(lst), fragment_size)]

    @staticmethod
    def crystal_ball(lst: List[Any]) -> Any:
        """
        Identifies the most frequently occurring element in a list.
        """
        if not lst:
            return None
        return Counter(lst).most_common(1)[0][0]

    @staticmethod
    def time_warp_list(lst: List[Any], positions: int) -> List[Any]:
        """
        Rotates the list by a given number of positions.
        """
        positions %= len(lst)
        return lst[-positions:] + lst[:-positions]

    @staticmethod
    def unique_artifacts(lst: List[Any]) -> List[Any]:
        """
        Gets elements that appear exactly once in the list.
        """
        frequency = Counter(lst)
        return [x for x in lst if frequency[x] == 1]

    @staticmethod
    def find_mystical_pairs(lst: List[int], target: int) -> List[tuple]:
        """
        Finds all pairs of numbers that sum to a specific target.
        """
        seen = set()
        pairs = []
        for num in lst:
            complement = target - num
            if complement in seen:
                pairs.append((complement, num))
            seen.add(num)
        return pairs

# Example usage of the functions in the script
if __name__ == "__main__":
    sample_list = [1, 2, 2, 3, 4, 4, 5, 5, 5]
    nested_list = [[1, 2], [3, 4], [5]]

    logging.info("Original List: %s", sample_list)
    logging.info("Without Duplicates: %s", ListWizard.magic_glimmer(sample_list))
    logging.info("Flattened Nested List: %s", ListWizard.teleport_together_list(nested_list))
    logging.info("List Intersection: %s", ListWizard.list_fusion([1, 2, 3], [2, 3, 4]))
    logging.info("Shuffled List: %s", ListWizard.enchant_shuffle(sample_list.copy()))  # Using copy to keep original
    logging.info("Sorted by Frequency: %s", ListWizard.sort_by_charm(sample_list))
    logging.info("Chunked List: %s", ListWizard.divide_list(sample_list, 2))
    logging.info("Most Frequent Element: %s", ListWizard.crystal_ball(sample_list))
    logging.info("Rotated List: %s", ListWizard.time_warp_list(sample_list, 2))
    logging.info("Unique Elements: %s", ListWizard.unique_artifacts(sample_list))
    logging.info("Pairs with Sum 6: %s", ListWizard.find_mystical_pairs(sample_list, 6))