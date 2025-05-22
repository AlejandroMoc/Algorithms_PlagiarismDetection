# Constants for placeholder characters
EMPTY = '\n'  # Used in encryption matrix
MARKER = '*'  # Used in decryption matrix

def change_direction(r, k, is_down):
    """
    Helper function to change direction in the rail matrix.
    
    Parameters:
    - r (int): Current row index.
    - k (int): Total number of rails (key).
    - is_down (bool): Current direction.
    
    Returns:
    - tuple: Updated row and direction.
    """
    if r == 0:
        is_down = True
    elif r == k - 1:
        is_down = False
    return r + (1 if is_down else -1), is_down

def encrypt_rail_fence(message, key):
    """
    Encrypts the given text using the Rail Fence Cipher method.
    
    Parameters:
    - message (str): The text to encrypt.
    - key (int): The number of rails to use in the cipher.

    Returns:
    - str: The encrypted text.
    """
    if not isinstance(message, str) or not isinstance(key, int):
        raise ValueError("Invalid input types. 'message' must be a string and 'key' must be an integer.")
    if key <= 0:
        raise ValueError("Key must be a positive integer.")
    
    # Initialize the rail matrix
    rail = [[EMPTY for _ in range(len(message))] for _ in range(key)]
    is_down = False
    r, c = 0, 0

    # Fill the rail matrix with characters in zig-zag fashion
    for char in message:
        rail[r][c] = char
        c += 1
        r, is_down = change_direction(r, key, is_down)

    # Collect the encrypted text
    result = [rail[i][j] for i in range(key) for j in range(len(message)) if rail[i][j] != EMPTY]
    return "".join(result)

def decrypt_rail_fence(encoded, key):
    """
    Decrypts the given cipher text using the Rail Fence Cipher method.
    
    Parameters:
    - encoded (str): The encrypted text to decrypt.
    - key (int): The number of rails to use in the cipher.

    Returns:
    - str: The decrypted text.
    """
    if not isinstance(encoded, str) or not isinstance(key, int):
        raise ValueError("Invalid input types. 'encoded' must be a string and 'key' must be an integer.")
    if key <= 0:
        raise ValueError("Key must be a positive integer.")
    
    # Initialize the rail matrix
    rail = [[EMPTY for _ in range(len(encoded))] for _ in range(key)]
    is_down = None
    r, c = 0, 0

    # Mark the pattern path in the rail matrix
    for _ in range(len(encoded)):
        rail[r][c] = MARKER
        c += 1
        r, is_down = change_direction(r, key, is_down)

    # Place characters in the marked positions
    idx = 0
    for i in range(key):
        for j in range(len(encoded)):
            if rail[i][j] == MARKER and idx < len(encoded):
                rail[i][j] = encoded[idx]
                idx += 1

    # Read the matrix in zig-zag fashion to decrypt
    result = []
    r, c = 0, 0
    for _ in range(len(encoded)):
        if rail[r][c] != MARKER:
            result.append(rail[r][c])
            c += 1
        r, is_down = change_direction(r, key, is_down)

    return "".join(result)