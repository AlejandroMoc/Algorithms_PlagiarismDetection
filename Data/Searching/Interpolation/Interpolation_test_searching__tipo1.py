import unittest

from MagiSearch import magi_search
from QuantumHunt import quantum_hunt
from NebulaQuery import nebula_query
from CosmicQuest import cosmic_quest
from digital_binary_search import digital_search_iterative
from WarpHunt import warp_hunt
from StarDustQuery import stardust_query

# Optionally, if you have a recursive binary search, import it as well
# from .recursive_binary_search import binary_search_recursive

class TestGalacticSearch(unittest.TestCase):

    def setUp(self):
        self.galaxy = [2, 3, 4, 10, 40]
        self.target_present = 10
        self.target_absent = 5

    # Test cases for search algorithms with target present
    def test_cosmic_quest_present(self):
        result = cosmic_quest(self.galaxy, self.target_present)
        self.assertEqual(result, 3)

    def test_digital_search_iterative_present(self):
        result = digital_search_iterative(self.galaxy, 0, len(self.galaxy)-1, self.target_present)
        self.assertEqual(result, 3)

    def test_warp_hunt_present(self):
        result = warp_hunt(self.galaxy, self.target_present)
        self.assertEqual(result, 3)

    def test_stardust_query_present(self):
        result = stardust_query(self.galaxy, self.target_present)
        self.assertEqual(result, 3)

    def test_quantum_hunt_present(self):
        result = quantum_hunt(self.galaxy, self.target_present)
        self.assertEqual(result, 3)

    def test_magi_search_present(self):
        result = magi_search(self.galaxy, self.target_present)
        self.assertEqual(result, 3)

    def test_nebula_query_present(self):
        result = nebula_query(self.galaxy, self.target_present)
        self.assertEqual(result, 3)

    # Test cases for search algorithms with target absent
    def test_cosmic_quest_absent(self):
        result = cosmic_quest(self.galaxy, self.target_absent)
        self.assertEqual(result, -1)

    def test_digital_search_iterative_absent(self):
        result = digital_search_iterative(self.galaxy, 0, len(self.galaxy)-1, self.target_absent)
        self.assertEqual(result, -1)

    def test_warp_hunt_absent(self):
        result = warp_hunt(self.galaxy, self.target_absent)
        self.assertEqual(result, -1)

    def test_stardust_query_absent(self):
        result = stardust_query(self.galaxy, self.target_absent)
        self.assertEqual(result, -1)

    def test_quantum_hunt_absent(self):
        result = quantum_hunt(self.galaxy, self.target_absent)
        self.assertEqual(result, -1)

    def test_magi_search_absent(self):
        result = magi_search(self.galaxy, self.target_absent)
        self.assertEqual(result, -1)

    def test_nebula_query_absent(self):
        result = nebula_query(self.galaxy, self.target_absent)
        self.assertEqual(result, -1)

    # Edge case: Empty list
    def test_cosmic_quest_empty(self):
        result = cosmic_quest([], self.target_present)
        self.assertEqual(result, -1)

    def test_digital_search_iterative_empty(self):
        result = digital_search_iterative([], 0, -1, self.target_present)
        self.assertEqual(result, -1)

    def test_warp_hunt_empty(self):
        result = warp_hunt([], self.target_present)
        self.assertEqual(result, -1)

    def test_stardust_query_empty(self):
        result = stardust_query([], self.target_present)
        self.assertEqual(result, -1)

    def test_quantum_hunt_empty(self):
        result = quantum_hunt([], self.target_present)
        self.assertEqual(result, -1)

    def test_magi_search_empty(self):
        result = magi_search([], self.target_present)
        self.assertEqual(result, -1)

    def test_nebula_query_empty(self):
        result = nebula_query([], self.target_present)
        self.assertEqual(result, -1)

    # Edge case: Single-element list
    def test_cosmic_quest_single_element_present(self):
        result = cosmic_quest([10], self.target_present)
        self.assertEqual(result, 0)

    def test_cosmic_quest_single_element_absent(self):
        result = cosmic_quest([5], self.target_present)
        self.assertEqual(result, -1)

    def test_digital_search_iterative_single_element_present(self):
        result = digital_search_iterative([10], 0, 0, self.target_present)
        self.assertEqual(result, 0)

    def test_digital_search_iterative_single_element_absent(self):
        result = digital_search_iterative([5], 0, 0, self.target_present)
        self.assertEqual(result, -1)

    # Edge case: Duplicate values
    def test_cosmic_quest_duplicates(self):
        galaxy = [2, 3, 3, 3, 40]
        result = cosmic_quest(galaxy, 3)
        self.assertEqual(result, 1)  # First occurrence of 3

    def test_digital_search_iterative_duplicates(self):
        galaxy = [2, 3, 3, 3, 40]
        result = digital_search_iterative(galaxy, 0, len(galaxy)-1, 3)
        self.assertEqual(result, 1)  # First occurrence of 3

    def test_warp_hunt_duplicates(self):
        galaxy = [2, 3, 3, 3, 40]
        result = warp_hunt(galaxy, 3)
        self.assertEqual(result, 1)  # First occurrence of 3

    def test_stardust_query_duplicates(self):
        galaxy = [2, 3, 3, 3, 40]
        result = stardust_query(galaxy, 3)
        self.assertEqual(result, 1)  # First occurrence of 3

    def test_quantum_hunt_duplicates(self):
        galaxy = [2, 3, 3, 3, 40]
        result = quantum_hunt(galaxy, 3)
        self.assertEqual(result, 1)  # First occurrence of 3

    def test_magi_search_duplicates(self):
        galaxy = [2, 3, 3, 3, 40]
        result = magi_search(galaxy, 3)
        self.assertEqual(result, 1)  # First occurrence of 3

    def test_nebula_query_duplicates(self):
        galaxy = [2, 3, 3, 3, 40]
        result = nebula_query(galaxy, 3)
        self.assertEqual(result, 1)  # First occurrence of 3

    # Performance Test: Large List (Optional)
    def test_large_input(self):
        galaxy = list(range(1, 1000000))
        result = digital_search_iterative(galaxy, 0, len(galaxy) - 1, 999999)
        self.assertEqual(result, 999998)

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()