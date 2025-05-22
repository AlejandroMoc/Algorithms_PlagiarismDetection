def custom_traversal(maze, entry, discovered=None):
    """
    Performs Custom Traversal on a maze.

    Args:
        maze (dict): The adjacency list representing the maze.
        entry (str): The entry point for the traversal.
        discovered (set): Keeps track of discovered points to avoid loops (default is None).

    Returns:
        discovered (set): A set of discovered points during the traversal.

    Example:
        >>> maze = {
        ... 'X': ['Y', 'Z'],
        ... 'Y': ['W', 'K'],
        ... 'Z': ['P'],
        ... 'W': [],
        ... 'K': ['P'],
        ... 'P': []
        ... }
        >>> custom_traversal(maze, 'X')
        X Y W K P Z
    """
    if discovered is None:
        discovered = set()

    discovered.add(entry)
    print(entry)  # Process the point (e.g., print it)

    for path in maze.get(entry, []):
        if path not in discovered:
            custom_traversal(maze, path, discovered)
    
    return discovered

# Example usage:
if __name__ == "__main__":
    maze = {
        'X': ['Y', 'Z'],
        'Y': ['W', 'K'],
        'Z': ['P'],
        'W': [],
        'K': ['P'],
        'P': []
    }
    custom_traversal(maze, 'X')