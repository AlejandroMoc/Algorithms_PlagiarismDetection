def add_numbers(var_a, var_b):
    """Suma dos números."""
    return var_a + var_b

def multiply_numbers(var_a, var_b):
    """Multiplica dos números."""
    return var_a * var_b

if __name__ == "__main__":
    result_add = add_numbers(5, 3)
    print(f"Suma: {result_add}")

    result_multiply = multiply_numbers(5, 3)
    print(f"Producto: {result_multiply}")