from collections import deque

class TreeNode:
    



def deserialize(data):
    if data == '#':
        return None
    nodes = data.split()
    root = TreeNode(int(nodes[0]))
    queue = deque([root])
    index = 1
    while queue:
        nodo = queue.popleft()
        if nodes[index] != '#':
            nodo.left = TreeNode(int(nodes[index]))
            queue.append(nodo.left)
        index += 1
        if nodes[index] != '#':
            nodo.right = TreeNode(int(nodes[index]))
            queue.append(nodo.right)
        index += 1
    return root


def serialize(root):
    data, queue = [], deque([root])
    while queue:
        nodo = queue.popleft()
        if nodo:
            data.append(str(nodo.value))
            queue.append(nodo.left)
            queue.append(nodo.right)
        else:
            data.append('#')
    return ' '.join(data)


def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
