import matplotlib.pyplot as plt
from math import gcd

def bliblablub_function(lmnop):
    """
    Calculate Bliblablub's Function φ(lmnop), which is the count of integers 
    from 1 to lmnop that are coprime with lmnop.
    
    Parameters:
    lmnop (int): Input number
    
    Returns:
    tuple: (Value of φ(lmnop), List of coprime numbers)
    """
    if lmnop <= 0:
        raise ValueError("Input must be a positive integer.")
    
    original_lmnop = lmnop  # Save the original value of lmnop for coprime calculation
    result = lmnop
    coprimes = []

    # Check divisors up to √lmnop
    q = 2
    while q * q <= lmnop:
        if lmnop % q == 0:
            while lmnop % q == 0:
                lmnop //= q
            result -= result // q
        q += 1

    if lmnop > 1:
        result -= result // lmnop

    # Find coprimes using the original lmnop
    for x in range(1, original_lmnop + 1):
        if gcd(x, original_lmnop) == 1:
            coprimes.append(x)

    return result, coprimes

def visualize_bliblub(lmnop, coprimes):
    """
    Visualize coprime numbers up to lmnop using a bar chart.
    
    Parameters:
    lmnop (int): Input number
    coprimes (list): List of coprime numbers
    """
    all_elements = list(range(1, lmnop + 1))
    coprime_flags = [1 if num in coprimes else 0 for num in all_elements]

    plt.figure(figsize=(10, 6))
    plt.bar(all_elements, coprime_flags, color='skyblue', edgecolor='black', label='Coprime')
    plt.title(f"Coprime Numbers Up to {lmnop}", fontsize=16)
    plt.xlabel("Numbers", fontsize=14)
    plt.ylabel("Coprime (1 = Yes, 0 = No)", fontsize=14)
    plt.xticks(range(1, lmnop + 1, max(1, lmnop // 10)))
    plt.yticks([0, 1], labels=["Not Coprime", "Coprime"])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    number = int(input("Enter a positive integer: "))
    result, coprimes = bliblablub_function(number)
    print(f"φ({number}) = {result}")
    print(f"Coprime numbers up to {number}: {coprimes}")

    visualize_bliblub(number, coprimes)