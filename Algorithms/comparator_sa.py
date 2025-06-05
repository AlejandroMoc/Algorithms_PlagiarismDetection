import os, tokenize
from io import BytesIO

# Leer archivo en modo binario
def read_code_bytes(file_content: str) -> bytes:
    return file_content.encode('utf-8')

# Tokenización con normalización de identificadores, números y cadenas
def preprocess_code(content):
    tokens = []
    name_map = {}
    name_index = 1

    reserved_keywords = {
        'def', 'return', 'if', 'else', 'elif', 'for', 'while', 'import', 'from',
        'as', 'try', 'except', 'finally', 'with', 'class', 'print', 'input', 'in',
        'not', 'and', 'or', 'is', 'lambda', 'pass', 'break', 'continue', 'True', 'False', 'None'
    }

    try:
        token_gen = tokenize.tokenize(BytesIO(content).readline)
        for token in token_gen:
            if token.type in [tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING, tokenize.ENDMARKER]:
                continue
            elif token.type == tokenize.NAME:
                if token.string in reserved_keywords:
                    tokens.append(token.string)
                else:
                    if token.string not in name_map:
                        name_map[token.string] = f'VAR{name_index}'
                        name_index += 1
                    tokens.append(name_map[token.string])
            elif token.type == tokenize.NUMBER:
                tokens.append('NUM')
            elif token.type == tokenize.STRING:
                tokens.append('STR')
            else:
                tokens.append(token.string)
    except tokenize.TokenError:
        pass
    return tokens

# Construir suffix array
def build_suffix_array(token_list):
    suffixes = [(token_list[i:], i) for i in range(len(token_list))]
    sorted_suffixes = sorted(suffixes)
    return [index for (_, index) in sorted_suffixes]

# Generar BWT (no cambia nada aquí)
def bwt_from_tokens(tokens, suffix_array):
    return [tokens[i - 1] if i != 0 else '$' for i in suffix_array]

# Función principal del comparador SA con mejor cálculo de similitud
def comparator_sa(code_1, code_2):
    # Convertir código a bytes y tokenizar
    bytes_1 = read_code_bytes(code_1)
    bytes_2 = read_code_bytes(code_2)

    tokens1 = preprocess_code(bytes_1)
    tokens2 = preprocess_code(bytes_2)

    # Fusionar tokens con separadores
    combined_tokens = tokens1 + ['#'] + tokens2 + ['$']
    separator_index = len(tokens1)

    # Construcción del SA
    suffix_array = build_suffix_array(combined_tokens)

    # Hallar la subcadena común más larga (LCS)
    max_lcp = 0
    best_pair = (0, 0)
    for i in range(1, len(suffix_array)):
        idx1 = suffix_array[i - 1]
        idx2 = suffix_array[i]
        # Comparar solo entre tokens de diferentes archivos
        if (idx1 < separator_index and idx2 > separator_index) or (idx1 > separator_index and idx2 < separator_index):
            lcp = 0
            while (idx1 + lcp < len(combined_tokens) and
                   idx2 + lcp < len(combined_tokens) and
                   combined_tokens[idx1 + lcp] == combined_tokens[idx2 + lcp] and
                   combined_tokens[idx1 + lcp] not in ['#', '$']):
                lcp += 1
            if lcp > max_lcp:
                max_lcp = lcp
                best_pair = (idx1, idx2)

    # Similaridad proporcional al promedio de tokens
    total_tokens = (len(tokens1) + len(tokens2)) / 2
    similarity = (max_lcp / total_tokens) * 100 if total_tokens > 0 else 0
    similarity = round(similarity, 2)

    return similarity

# Prueba
def main():
    file_a = "Data/testcase1.py"
    file_b = "Data/testcase7.py"

    with open(file_a, 'r', encoding='utf-8') as f:
        code1 = f.read()
    with open(file_b, 'r', encoding='utf-8') as f:
        code2 = f.read()

    resultado = comparator_sa(code1, code2)
    print(f"📊 Similitud Suffix Array (global, tokens normalizados): {resultado:.2f}%")

if __name__ == '__main__':
    main()
