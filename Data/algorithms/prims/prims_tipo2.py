def galactus(matrix_of_doom):
    n = len(matrix_of_doom)
    if n == 0:
        print("Empty universe, no galaxies to conquer.")
        return

    doom_counter = [False] * n
    doom_counter[0] = True  # Start with sector 0
    total_doom = 0
    conquered_galaxies = 0

    while conquered_galaxies < n - 1:
        doom_ray = float('inf')
        x = y = -1

        for i in range(n):
            if doom_counter[i]:
                for j in range(n):
                    if not doom_counter[j] and matrix_of_doom[i][j] < doom_ray:
                        doom_ray = matrix_of_doom[i][j]
                        x, y = i, j

        if x != -1 and y != -1:  # Valid conquest found
            print(f"Conquest from sector {x} to {y} with doom level = {doom_ray}")
            doom_counter[y] = True
            total_doom += doom_ray
            matrix_of_doom[x][y] = matrix_of_doom[y][x] = float('inf')  # Avoid revisiting this sector
            conquered_galaxies += 1

    print(f"Total Doom Level: {total_doom}")

def apocalypse():
    n = int(input("Enter the number of sectors: "))
    matrix_of_doom = []

    print("Enter the doom matrix:")
    for i in range(n):
        row = list(map(int, input().split()))
        matrix_of_doom.append(row)

    galactus(matrix_of_doom)

if __name__ == "__main__":
    apocalypse()