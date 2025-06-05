class TreeNode:
    


def is_balanced(root):
    def check_balance(nodo):
        if not nodo:
            return 0
        left_height = check_balance(nodo.left)
        if left_height == -1:
            return -1
        right_height = check_balance(nodo.right)
        if right_height == -1 or abs(left_height - right_height) > 1:
            return -1
        return 1 + max(left_height, right_height)
    return check_balance(root) != -1


def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
