class TreeNode:
    

    
    depth(root)
    return diameter


def depth(nodo):
        nonlocal diameter
        if not nodo:
            return 0
        left_depth = depth(nodo.left)
        right_depth = depth(nodo.right)
        diameter = max(diameter, left_depth + right_depth)
        return 1 + max(left_depth, right_depth)


def tree_diameter(root):
    diameter = 0


def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
