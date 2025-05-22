def g(lst, x, y):
    # Base condition
    if x == y:
        return 0
    
    mini = float('inf')
    
    # Partitioning loop
    for z in range(x, y):
        result = g(lst, x, z) + g(lst, z + 1, y) + lst[x - 1] * lst[z] * lst[y]
        mini = min(mini, result)
    
    return mini

def matrix_operation(lst, N):
    x = 1
    y = N - 1
    return g(lst, x, y)

def main():
    lst = [5, 10, 15, 20, 25]
    n = len(lst)
    print("The minimum number of operations are", matrix_operation(lst, n))

if __name__ == "__main__":
    main()