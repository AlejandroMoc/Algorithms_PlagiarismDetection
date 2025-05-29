import unittest
from Circular_Linked_List import insert_end as add_end_cll, delete_node as remove_node_cll, display as show_cll
from Doubly_Linked_List import insert_end as add_end_dll, delete_node as remove_node_dll, display as show_dll
from Singly_Linked_List import insert_end as add_end_sll, delete_node as remove_node_sll, display as show_sll

class TestCustomLists(unittest.TestCase):

    def add_elements(self, add_function, head, elements):
        """Helper method to add multiple elements at the end of the list."""
        for element in elements:
            head = add_function(head, element)
        return head

    def test_circular_linked_list(self):
        head = None
        head = self.add_elements(add_end_cll, head, [1, 2, 3])
        self.assertEqual(show_cll(head), [1, 2, 3])
        head = remove_node_cll(head, 2)
        self.assertEqual(show_cll(head), [1, 3])

        # Edge case: Removing from an empty list
        head = remove_node_cll(head, 5)
        self.assertEqual(show_cll(head), [1, 3])

        # Edge case: Removing the head node
        head = remove_node_cll(head, 1)
        self.assertEqual(show_cll(head), [3])

    def test_doubly_linked_list(self):
        head = None
        head = self.add_elements(add_end_dll, head, [1, 2, 3])
        self.assertEqual(show_dll(head), [1, 2, 3])
        head = remove_node_dll(head, 2)
        self.assertEqual(show_dll(head), [1, 3])

        # Edge case: Removing from an empty list
        head = remove_node_dll(head, 5)
        self.assertEqual(show_dll(head), [1, 3])

        # Edge case: Removing the head node
        head = remove_node_dll(head, 1)
        self.assertEqual(show_dll(head), [3])

    def test_singly_linked_list(self):
        head = None
        head = self.add_elements(add_end_sll, head, [1, 2, 3])
        self.assertEqual(show_sll(head), [1, 2, 3])
        head = remove_node_sll(head, 2)
        self.assertEqual(show_sll(head), [1, 3])

        # Edge case: Removing from an empty list
        head = remove_node_sll(head, 5)
        self.assertEqual(show_sll(head), [1, 3])

        # Edge case: Removing the head node
        head = remove_node_sll(head, 1)
        self.assertEqual(show_sll(head), [3])

if __name__ == '__main__':
    unittest.main()