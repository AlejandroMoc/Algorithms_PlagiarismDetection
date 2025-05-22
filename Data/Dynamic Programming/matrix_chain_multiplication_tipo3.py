def matrix_chain_order(arr, i, j):
    if i == j:
        return 0
    
    min_operations = float('inf')
    
    for k in range(i, j):
        operations = matrix_chain_order(arr, i, k) + matrix_chain_order(arr, k + 1, j) + arr[i - 1] * arr[k] * arr[j]
        min_operations = min(min_operations, operations)
    
    return min_operations

def matrix_multiplication(arr):
    start_index = 1
    end_index = len(arr) - 1
    return matrix_chain_order(arr, start_index, end_index)

def main():
    arr = [5, 10, 15, 20, 25]
    n = len(arr)
    print("The minimum number of operations are", matrix_multiplication(arr))

if __name__ == "__main__":
    main()