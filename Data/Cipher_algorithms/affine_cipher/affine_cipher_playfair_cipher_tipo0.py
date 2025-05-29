# Constants
SPACER_CHAR = 'X'
REPLACER_CHAR = 'I'

def generate_coded_table(codeword):
    """
    Generates a 5x5 coded table for the Cryptify cipher.
    
    Parameters:
    - codeword (str): The keyword to generate the table from.
    
    Returns:
    - list: A 5x5 matrix of characters as the coded table.
    """
    # Remove duplicate characters in the codeword, preserving the order
    codeword = ''.join(sorted(set(codeword), key=lambda x: codeword.index(x)))
    # Fill the table with A-Z, excluding 'J' and already included codeword characters
    codeword += ''.join(chr(i) for i in range(65, 91) if chr(i) not in codeword and chr(i) != 'J')
    return [list(codeword[i:i+5]) for i in range(0, 25, 5)]

def preprocess_message(message):
    """
    Preprocesses the plaintext by converting to uppercase, replacing 'J' with 'I',
    and adding spacer characters where needed.
    
    Parameters:
    - message (str): The plaintext message to preprocess.
    
    Returns:
    - str: The processed message with spacers.
    """
    message = message.upper().replace('J', REPLACER_CHAR)
    processed_message = ""
    i = 0
    while i < len(message):
        processed_message += message[i]
        # Add spacer if two consecutive letters are the same
        if i + 1 < len(message) and message[i] == message[i + 1]:
            processed_message += SPACER_CHAR
        elif i + 1 < len(message):
            processed_message += message[i + 1]
            i += 1
        i += 1
    # Append spacer if the length is odd
    if len(processed_message) % 2 != 0:
        processed_message += SPACER_CHAR
    return processed_message

def locate_position(char, coded_table):
    """
    Finds the position of a character in the coded table.
    
    Parameters:
    - char (str): The character to locate in the table.
    - coded_table (list): The 5x5 matrix coded table.
    
    Returns:
    - tuple: The row and column indices of the character.
    """
    for i, row in enumerate(coded_table):
        if char in row:
            return i, row.index(char)
    return None

def cryptify_encrypt(plaintext, codeword):
    """
    Encrypts the plaintext using the Cryptify cipher with the provided codeword.
    
    Parameters:
    - plaintext (str): The text to encrypt.
    - codeword (str): The codeword for generating the cipher's coded table.
    
    Returns:
    - str: The encrypted ciphertext.
    """
    coded_table = generate_coded_table(codeword)
    plaintext = preprocess_message(plaintext)
    ciphertext = ""
    for i in range(0, len(plaintext), 2):
        row1, col1 = locate_position(plaintext[i], coded_table)
        row2, col2 = locate_position(plaintext[i + 1], coded_table)
        if row1 == row2:
            # Same row: shift right
            ciphertext += coded_table[row1][(col1 + 1) % 5]
            ciphertext += coded_table[row2][(col2 + 1) % 5]
        elif col1 == col2:
            # Same column: shift down
            ciphertext += coded_table[(row1 + 1) % 5][col1]
            ciphertext += coded_table[(row2 + 1) % 5][col2]
        else:
            # Rectangle swap
            ciphertext += coded_table[row1][col2]
            ciphertext += coded_table[row2][col1]
    return ciphertext

def cryptify_decrypt(ciphertext, codeword):
    """
    Decrypts the ciphertext using the Cryptify cipher with the provided codeword.
    
    Parameters:
    - ciphertext (str): The text to decrypt.
    - codeword (str): The codeword for generating the cipher's coded table.
    
    Returns:
    - str: The decrypted plaintext.
    """
    coded_table = generate_coded_table(codeword)
    plaintext = ""
    for i in range(0, len(ciphertext), 2):
        row1, col1 = locate_position(ciphertext[i], coded_table)
        row2, col2 = locate_position(ciphertext[i + 1], coded_table)
        if row1 == row2:
            # Same row: shift left
            plaintext += coded_table[row1][(col1 - 1) % 5]
            plaintext += coded_table[row2][(col2 - 1) % 5]
        elif col1 == col2:
            # Same column: shift up
            plaintext += coded_table[(row1 - 1) % 5][col1]
            plaintext += coded_table[(row2 - 1) % 5][col2]
        else:
            # Rectangle swap
            plaintext += coded_table[row1][col2]
            plaintext += coded_table[row2][col1]
    return plaintext