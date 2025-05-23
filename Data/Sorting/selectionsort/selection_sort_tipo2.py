def quirky_sort(lst):
    n = len(lst)
    for i in range(n):
        idx1 = i
        for j in range(i + 1, n):
            if lst[j] < lst[idx1]:
                idx1 = j
        lst[i], lst[idx1] = lst[idx1], lst[i]
    return lst

if __name__ == "__main__":
    # Input: space-separated integers
    input_data = input("Enter numbers to sort, separated by spaces: ")
    lst = list(map(int, input_data.split()))
    
    sorted_lst = quirky_sort(lst)
    print("Sorted array:", sorted_lst)