import heapq

class Noodle:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to end
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def noodles_algo(pasta, noodle_start, noodle_end, allow_twist=False):
    noodle_start = Noodle(noodle_start)
    noodle_end = Noodle(noodle_end)

    open_dish = []
    closed_dish = set()
    open_map = {}  # Dictionary to track nodes and their 'g' values

    heapq.heappush(open_dish, noodle_start)
    open_map[noodle_start.position] = noodle_start

    while open_dish:
        current_noodle = heapq.heappop(open_dish)
        closed_dish.add(current_noodle.position)

        if current_noodle == noodle_end:
            # Reconstruct path
            path = []
            while current_noodle:
                path.append(current_noodle.position)
                current_noodle = current_noodle.parent
            return path[::-1]  # Return path in correct order

        ingredients = []
        spices = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        if allow_twist:
            # Include diagonal movements (top-left, top-right, bottom-left, bottom-right)
            spices += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Generate children (neighbors)
        for new_position in spices:
            noodle_position = (current_noodle.position[0] + new_position[0], current_noodle.position[1] + new_position[1])

            # Skip out-of-bounds or blocked cells (value != 0 means wall)
            if noodle_position[0] > (len(pasta) - 1) or noodle_position[0] < 0 or noodle_position[1] > (len(pasta[0]) - 1) or noodle_position[1] < 0:
                continue
            if pasta[noodle_position[0]][noodle_position[1]] != 0:
                continue

            new_noodle = Noodle(noodle_position, current_noodle)
            ingredients.append(new_noodle)

        for ingredient in ingredients:
            if ingredient.position in closed_dish:
                continue  # Ignore already evaluated nodes

            ingredient.g = current_noodle.g + 1  # g value is distance from start
            # Using Manhattan distance as heuristic (for non-diagonal movement)
            ingredient.h = abs(ingredient.position[0] - noodle_end.position[0]) + abs(ingredient.position[1] - noodle_end.position[1])
            ingredient.f = ingredient.g + ingredient.h

            # Check if this node should be added to open_dish or skipped
            if ingredient.position not in open_map or ingredient.g < open_map[ingredient.position].g:
                open_map[ingredient.position] = ingredient
                heapq.heappush(open_dish, ingredient)

    return None  # Return None if no path is found

# Example usage:
pasta = [
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0]
]

noodle_start = (0, 0)
noodle_end = (4, 5)

# Calling the A* algorithm with diagonal movement allowed
path = noodles_algo(pasta, noodle_start, noodle_end, allow_twist=True)
print("Path:", path)