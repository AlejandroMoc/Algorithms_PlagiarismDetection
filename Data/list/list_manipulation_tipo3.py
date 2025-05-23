import logging
from collections import Counter
from dataclasses import dataclass
from typing import List, Any

logging.basicConfig(level=logging.INFO)

@dataclass
class ManipulatorOfLists:
    
    @staticmethod
    def eliminate_duplicates(lst: List[Any]) -> List[Any]:
        seen_elements = set()
        return [elem for elem in lst if not (elem in seen_elements or seen_elements.add(elem))]
    
    @staticmethod
    def flatten_nested(lst_nested: List[List[Any]]) -> List[Any]:
        return [item for sublist in lst_nested for item in sublist]
    
    @staticmethod
    def intersection_of_lists(lst1: List[Any], lst2: List[Any]) -> List[Any]:
        return list(set(lst1) & set(lst2))
    
    @staticmethod
    def shuffle_randomly(lst: List[Any]) -> List[Any]:
        import random
        random.shuffle(lst)
        return lst
    
    @staticmethod
    def sort_by_occurrence(lst: List[Any]) -> List[Any]:
        frequency_counter = Counter(lst)
        return sorted(lst, key=lambda x: frequency_counter[x], reverse=True)
    
    @staticmethod
    def split_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    
    @staticmethod
    def most_occuring_element(lst: List[Any]) -> Any:
        if not lst:
            return None
        return Counter(lst).most_common(1)[0][0]
    
    @staticmethod
    def rotate_list(lst: List[Any], positions: int) -> List[Any]:
        positions %= len(lst)
        return lst[-positions:] + lst[:-positions]
    
    @staticmethod
    def unique_occurrences(lst: List[Any]) -> List[Any]:
        frequency_counter = Counter(lst)
        return [x for x in lst if frequency_counter[x] == 1]
    
    @staticmethod
    def find_sum_pairs(lst: List[int], target: int) -> List[tuple]:
        seen_nums = set()
        pairs_found = []
        for num in lst:
            complement = target - num
            if complement in seen_nums:
                pairs_found.append((complement, num))
            seen_nums.add(num)
        return pairs_found

if __name__ == "__main__":
    sample_lst = [1, 2, 2, 3, 4, 4, 5, 5, 5]
    nested_lst = [[1, 2], [3, 4], [5]]
    
    logging.info("Original List: %s", sample_lst)
    logging.info("Without Duplicates: %s", ManipulatorOfLists.eliminate_duplicates(sample_lst))
    logging.info("Flattened Nested List: %s", ManipulatorOfLists.flatten_nested(nested_lst))
    logging.info("List Intersection: %s", ManipulatorOfLists.intersection_of_lists([1, 2, 3], [2, 3, 4]))
    logging.info("Shuffled List: %s", ManipulatorOfLists.shuffle_randomly(sample_lst.copy()))
    logging.info("Sorted by Frequency: %s", ManipulatorOfLists.sort_by_occurrence(sample_lst))
    logging.info("Chunked List: %s", ManipulatorOfLists.split_list(sample_lst, 2))
    logging.info("Most Frequent Element: %s", ManipulatorOfLists.most_occuring_element(sample_lst))
    logging.info("Rotated List: %s", ManipulatorOfLists.rotate_list(sample_lst, 2))
    logging.info("Unique Elements: %s", ManipulatorOfLists.unique_occurrences(sample_lst))
    logging.info("Pairs with Sum 6: %s", ManipulatorOfLists.find_sum_pairs(sample_lst, 6))