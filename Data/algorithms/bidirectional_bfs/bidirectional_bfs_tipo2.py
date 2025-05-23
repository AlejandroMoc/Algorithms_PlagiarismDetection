from collections import deque

def zigzag_traversal(mosaic, origin, destination):
    '''
    Performs a zigzag traversal on a complex pattern and returns the shortest route between two points.
    Simultaneously utilizes two breadth-first explorations (BFE) from the origin and destination points.
    Upon encountering each other, the route is reconstructed by retracing steps back to the origin and destination points.

    Args:
        mosaic (dict): A dictionary representing a complex pattern where keys are point identifiers and values
                        are lists of adjacent points.
        origin: The starting point in the pattern for the traversal. Corresponds to a key in the `mosaic` dictionary.
        destination: The destination point in the pattern for the traversal. Corresponds to a key in the `mosaic` dictionary.

    Returns:
        list: A list representing the shortest route from `origin` to `destination`. If no route exists, returns `None`.
    '''
    if not isinstance(mosaic, dict):
        raise TypeError("Mosaic should be a dictionary with points as keys and lists of adjacent points as values.")
    if origin == destination:
        return [origin]
    if origin not in mosaic or destination not in mosaic:
        raise ValueError(f"Origin ({origin}) and destination ({destination}) points must be present in the mosaic.")

    queue_origin = deque([origin])
    queue_destination = deque([destination])

    visited_origin = {origin}
    visited_destination = {destination}

    predecessors_origin = {origin: None}
    predecessors_destination = {destination: None}

    while queue_origin and queue_destination:
        # Perform BFE from the origin side
        route = breadth_first_exploration(mosaic, visited_origin, queue_origin, predecessors_origin, visited_destination)
        if route:
            return route

        # Perform BFE from the destination side
        route = breadth_first_exploration(mosaic, visited_destination, queue_destination, predecessors_destination, visited_origin)
        if route:
            return route

    return None  # No route found

def breadth_first_exploration(mosaic, visited, queue, predecessors, other_visited):
    '''
    Breadth-first exploration from the current point to find neighbors and check for meeting points with the opposite exploration.

    Args:
        mosaic (dict): The pattern represented as a dictionary where each key is a point and its value is a list
                        of adjacent points.
        visited (set): A set of points already visited in this direction of the exploration.
        queue (deque): The BFE queue holding points to explore.
        predecessors (dict): A dictionary mapping each visited point to its predecessor point.
        other_visited (set): A set of points visited by the BFE running in the opposite direction.

    Returns:
        list or None: Returns the complete route if a meeting point is found, otherwise `None`.
    '''
    current_point = queue.popleft()
    for neighbor in mosaic[current_point]:
        if neighbor not in visited:
            predecessors[neighbor] = current_point
            visited.add(neighbor)
            queue.append(neighbor)

            if neighbor in other_visited:
                # Reconstruct the route from origin to destination through the meeting point
                return reconstruct_route(predecessors, current_point, neighbor)
    return None

def reconstruct_route(predecessors, origin_point, meeting_point):
    '''
    Reconstruct the route from the origin point to the meeting point by utilizing the predecessor points.

    Args:
        predecessors (dict): A dictionary where each key is a point and the value is the predecessor point in the route.
        origin_point: The starting point of the traversal.
        meeting_point: The point where the bidirectional traversal from origin and destination meets.

    Returns:
        list: A list of points representing the route from origin_point to the meeting_point.
    '''
    route = []
    # Backtrack from the meeting point to the origin point
    point = meeting_point
    while point is not None:
        route.append(point)
        point = predecessors[point]
    
    route.reverse()
    return route

# Example usage:
if __name__ == "__main__":
    mosaic = {
        'X': ['Y', 'Z'],
        'Y': ['X', 'A', 'B'],
        'Z': ['X', 'C'],
        'A': ['Y'],
        'B': ['Y', 'C'],
        'C': ['Z', 'B']
    }

    origin, destination = 'X', 'C'
    route = zigzag_traversal(mosaic, origin, destination)
    print(f"Shortest route from {origin} to {destination}: {route}")