def recursive_traversal(maze, entry, visited=None):
    """
    Performs Recursive Traversal on a maze represented as a dictionary.

    Args:
        maze (dict): The maze represented as a dictionary.
        entry (str): The starting point for the traversal.
        visited (set): Keeps track of visited points to avoid cycles (default is None).

    Returns:
        list: The list of visited points in order.

    Example:
        >>> maze = {
        ... 'X': ['Y', 'Z'],
        ... 'Y': ['W', 'R'],
        ... 'Z': ['Q'],
        ... 'W': [],
        ... 'R': ['Q'],
        ... 'Q': [],
        ... }
        >>> recursive_traversal(maze, 'X')
        X Y W R Q Z
    """
    if not isinstance(maze, dict):
        raise TypeError("Maze should be a dictionary with points as keys and lists of connections as values.")
    
    if entry not in maze:
        raise ValueError(f"Entry point {entry} not found in the maze.")

    if visited is None:
        visited = set()

    visited.add(entry)
    print(entry, end=' ')

    for connection in maze[entry]:
        if connection not in visited:
            recursive_traversal(maze, connection, visited)

    return list(visited)  # Return the order of visited points

# Example usage
if __name__ == "__main__":
    maze = {
        'X': ['Y', 'Z'],
        'Y': ['W', 'R'],
        'Z': ['Q'],
        'W': [],
        'R': ['Q'],
        'Q': []
    }
    visited_points = recursive_traversal(maze, 'X')  # Output: X Y W R Q Z
    print("\nVisited Points:", visited_points)