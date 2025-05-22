import matplotlib.pyplot as plt
from math import gcd

def totient_function(n):
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    
    original_n = n
    result = n
    coprimes = []

    factor = 2
    while factor * factor <= n:
        if n % factor == 0:
            while n % factor == 0:
                n //= factor
            result -= result // factor
        factor += 1

    if n > 1:
        result -= result // n

    for i in range(1, original_n + 1):
        if gcd(i, original_n) == 1:
            coprimes.append(i)

    return result, coprimes

def visualize_coprimes(n, coprimes):
    all_numbers = list(range(1, n + 1))
    coprime_flags = [1 if num in coprimes else 0 for num in all_numbers]

    plt.figure(figsize=(10, 6))
    plt.bar(all_numbers, coprime_flags, color='skyblue', edgecolor='black', label='Coprime')
    plt.title(f"Coprime Numbers Up to {n}", fontsize=16)
    plt.xlabel("Numbers", fontsize=14)
    plt.ylabel("Coprime (1 = Yes, 0 = No)", fontsize=14)
    plt.xticks(range(1, n + 1, max(1, n // 10)))
    plt.yticks([0, 1], labels=["Not Coprime", "Coprime"])
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    num = int(input("Enter a positive integer: "))
    result, coprimes = totient_function(num)
    print(f"φ({num}) = {result}")
    print(f"Coprime numbers up to {num}: {coprimes}")

    visualize_coprimes(num, coprimes)