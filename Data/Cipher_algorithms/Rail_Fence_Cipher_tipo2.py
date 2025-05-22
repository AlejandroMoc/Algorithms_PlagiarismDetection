# Constants for placeholder characters
BLANKY_CHAR = '\n'  # Used in encryption matrix
POINTER_CHAR = '*'  # Used in decryption matrix

def shift_orientation(row, key, go_down):
    """
    Helper function to toggle direction in the rail matrix.
    
    Parameters:
    - row (int): Current row index.
    - key (int): Total number of rails (key).
    - go_down (bool): Current direction.
    
    Returns:
    - tuple: Updated row and direction.
    """
    if row == 0:
        go_down = True
    elif row == key - 1:
        go_down = False
    return row + (1 if go_down else -1), go_down

def zigzag_fence_secret_encode(text, key):
    """
    Encrypts the given text using the Rail Fence Cipher method.
    
    Parameters:
    - text (str): The text to encrypt.
    - key (int): The number of rails to use in the cipher.

    Returns:
    - str: The encrypted text.
    """
    if not isinstance(text, str) or not isinstance(key, int):
        raise ValueError("Invalid input types. 'text' must be a string and 'key' must be an integer.")
    if key <= 0:
        raise ValueError("Key must be a positive integer.")
    
    # Initialize the rail matrix
    wire = [[BLANKY_CHAR for _ in range(len(text))] for _ in range(key)]
    go_down = False
    row, col = 0, 0

    # Fill the rail matrix with characters in zig-zag fashion
    for char in text:
        wire[row][col] = char
        col += 1
        row, go_down = shift_orientation(row, key, go_down)

    # Collect the encrypted text
    result = [wire[i][j] for i in range(key) for j in range(len(text)) if wire[i][j] != BLANKY_CHAR]
    return "".join(result)


def zigzag_fence_secret_decode(secret, key):
    """
    Decrypts the given cipher text using the Rail Fence Cipher method.
    
    Parameters:
    - secret (str): The encrypted text to decrypt.
    - key (int): The number of rails to use in the cipher.

    Returns:
    - str: The decrypted text.
    """
    if not isinstance(secret, str) or not isinstance(key, int):
        raise ValueError("Invalid input types. 'secret' must be a string and 'key' must be an integer.")
    if key <= 0:
        raise ValueError("Key must be a positive integer.")
    
    # Initialize the rail matrix
    wire = [[BLANKY_CHAR for _ in range(len(secret))] for _ in range(key)]
    go_down = None
    row, col = 0, 0

    # Mark the pattern path in the rail matrix
    for _ in range(len(secret)):
        wire[row][col] = POINTER_CHAR
        col += 1
        row, go_down = shift_orientation(row, key, go_down)

    # Place characters in the marked positions
    index = 0
    for i in range(key):
        for j in range(len(secret)):
            if wire[i][j] == POINTER_CHAR and index < len(secret):
                wire[i][j] = secret[index]
                index += 1

    # Read the matrix in zig-zag fashion to decrypt
    result = []
    row, col = 0, 0
    for _ in range(len(secret)):
        if wire[row][col] != POINTER_CHAR:
            result.append(wire[row][col])
            col += 1
        row, go_down = shift_orientation(row, key, go_down)

    return "".join(result)