from io import BytesIO
import difflib, os, glob, itertools, tokenize

def read_code_bytes(file_content: str) -> bytes:
    return file_content.encode('utf-8')

#Diferencias texto
def compare_contents(content_1, content_2):
    differences = difflib.ndiff(content_1, content_2)
    return '\n'.join(differences)

#Similitud con ratio()
def compare_similarity_ratio(tokens_1, tokens_2):
    matcher = difflib.SequenceMatcher(None, tokens_1, tokens_2)
    return matcher.ratio() * 100

#Preprocesamiento léxico
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
        #Obtener tokens de archivo; iterar por token y obtener tipo
        token_gen = tokenize.tokenize(BytesIO(content).readline)
        for token in token_gen:

            #Si el token es comentario, salto de línea o fin de archivo, saltars
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

#Comparador preprocesado
def compare_preprocessed(code_1: str, code_2: str):
    raw_1 = read_code_bytes(code_1)
    raw_2 = read_code_bytes(code_2)

    tokens_1 = preprocess_code(raw_1)
    tokens_2 = preprocess_code(raw_2)

    similarity = compare_similarity_ratio(tokens_1, tokens_2)
    result_comparison = compare_contents(tokens_1, tokens_2)

    output = f"Similitud (preprocesado): {similarity:.2f}%\n"
    output += "Diferencias:\n" + result_comparison + "\n"
    return similarity, output

#Comparador texto plano
def compare_plain(content_1: str, content_2: str):

    similarity = compare_similarity_ratio(content_1, content_2)
    result_comparison = compare_contents(content_1, content_2)

    output = f"Similitud (texto llano): {similarity:.2f}%\n"
    output += "Diferencias:\n" + result_comparison + "\n"
    return similarity, output

#Función principal del difflib
def comparator_difflib(file_a: str, file_b: str):
    similarity_preprocessed, result_preprocessed = compare_preprocessed(file_a, file_b)
    similarity_plain, result_plain = compare_plain(file_a, file_b)
    return (similarity_preprocessed, result_preprocessed, similarity_plain, result_plain)

#Ejecución principal
def main():
    file_a = "Data/testcase1.py"
    file_b = "Data/testcase7.py"

    resultado = comparator_difflib(file_a, file_b)
    print(resultado)

if __name__ == '__main__':
    main()