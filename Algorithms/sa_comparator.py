#Comparador de código usando Suffix Array
## A01736339 - Jacqueline Villa Asencio
## A01736671 - José Juan Irene Cervantes
## A01736346 - Augusto Gómez Maxil
## A01736353 - Alejandro Daniel Moctezuma Cruz

import os, glob
from io import BytesIO
import tokenize

# Leer archivo en modo binario
def read_file_raw(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

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

# Generar BWT a partir del suffix array
def bwt_from_tokens(tokens, suffix_array):
    return [tokens[i - 1] if i != 0 else '$' for i in suffix_array]

## PROGRAMA PRINCIPAL

def main():
    subCompa = "sub7"  # Subcarpeta a analizar

    print("== Comparador de Python usando Suffix Array + BWT ==")

    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, 'Data', 'Python', subCompa)
    archivos = sorted(glob.glob(os.path.join(data_dir, '*.py')))
    print(f"Se encontraron {len(archivos)} archivos para analizar.\n")

    if len(archivos) != 2:
        print("⚠️ Deben existir exactamente 2 archivos para comparar.")
        return

    file1, file2 = archivos

    # Tokenización de ambos archivos
    raw1 = read_file_raw(file1)
    raw2 = read_file_raw(file2)

    tokens1 = preprocess_code(raw1)
    tokens2 = preprocess_code(raw2)

    # Unir tokens para análisis con separadores
    combined_tokens = tokens1 + ['#'] + tokens2 + ['$']
    separator_index = len(tokens1)

    # Crear suffix array
    suffix_array = build_suffix_array(combined_tokens)

    # Buscar subcadena común más larga
    max_lcp = 0
    best_pair = (0, 0)

    for i in range(1, len(suffix_array)):
        idx1 = suffix_array[i - 1]
        idx2 = suffix_array[i]
        if (idx1 < separator_index and idx2 > separator_index) or (idx1 > separator_index and idx2 < separator_index):
            lcp = 0
            while (idx1 + lcp < len(combined_tokens) and
                   idx2 + lcp < len(combined_tokens) and
                   combined_tokens[idx1 + lcp] == combined_tokens[idx2 + lcp]):
                lcp += 1
            if lcp > max_lcp:
                max_lcp = lcp
                best_pair = (idx1, idx2)

    longest_common_substring = combined_tokens[best_pair[0]:best_pair[0] + max_lcp]
    similarity = (max_lcp / min(len(tokens1), len(tokens2))) * 100 if min(len(tokens1), len(tokens2)) > 0 else 0

    # Generar BWT
    bwt_result = bwt_from_tokens(combined_tokens, suffix_array)

    # Escribir resultados
    with open("resultado_comparador_sa.txt", "w", encoding="utf-8") as report:
        report.write(f"Archivo 1: {os.path.basename(file1)}\n")
        report.write(f"Tokens ({len(tokens1)}):\n")
        report.write(' '.join(tokens1) + "\n")
        report.write("=" * 60 + "\n")

        report.write(f"Archivo 2: {os.path.basename(file2)}\n")
        report.write(f"Tokens ({len(tokens2)}):\n")
        report.write(' '.join(tokens2) + "\n")
        report.write("=" * 60 + "\n")

        report.write(f"\nSimilitud (Suffix Array): {similarity:.2f}%\n")
        report.write("Subcadena común más larga:\n")
        report.write(' '.join(longest_common_substring) + "\n")

        report.write("\nTransformación BWT del texto combinado:\n")
        report.write(' '.join(bwt_result) + "\n")

    print("✅ Comparación completada.")
    print(f"📄 Resultado guardado en 'resultado_comparador_sa.txt'")
    print(f"🔗 Subcadena común más larga: {' '.join(longest_common_substring)}")
    print(f"📊 Porcentaje de similitud: {similarity:.2f}%")
    print(f"🌀 BWT generado: {' '.join(bwt_result)}")

if __name__ == '__main__':
    main()
