from collections import deque

class TreeNode:
    






def max_depth(root):
    if not root:
        return 0
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    return 1 + max(left_depth, right_depth)


def level_order_traversal(root):
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            nodo = queue.popleft()
            level.append(nodo.value)
            if nodo.left:
                queue.append(nodo.left)
            if nodo.right:
                queue.append(nodo.right)
        result.append(level)
    return result


def postorder_traversal(root):
    return postorder_traversal(root.left) + postorder_traversal(root.right) + [root.value] if root else []


def preorder_traversal(root):
    return [root.value] + preorder_traversal(root.left) + preorder_traversal(root.right) if root else []


def inorder_traversal(root):
    return inorder_traversal(root.left) + [root.value] + inorder_traversal(root.right) if root else []


def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
