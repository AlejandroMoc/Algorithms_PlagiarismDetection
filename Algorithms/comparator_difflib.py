from io import BytesIO
import difflib, os, glob, itertools, tokenize

#Lectura de archivos
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()

def read_file_raw(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

#Diferencias texto
def compare_contents(content1, content2):
    differences = difflib.ndiff(content1, content2)
    return '\n'.join(differences)

#Similitud con ratio()
def compare_similarity_ratio(tokens1, tokens2):
    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
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
def compare_preprocessed(code1, code2):
    raw1 = read_file_raw(code1)
    raw2 = read_file_raw(code2)

    tokens1 = preprocess_code(raw1)
    tokens2 = preprocess_code(raw2)

    similarity = compare_similarity_ratio(tokens1, tokens2)
    result_comparison = compare_contents(tokens1, tokens2)

    output = f"Similitud (preprocesado): {similarity:.2f}%\n"
    output += "Diferencias:\n" + result_comparison + "\n"
    return output

#Comparador texto plano
def compare_plain(code1, code2):
    content1 = read_file(code1)
    content2 = read_file(code2)

    similarity = compare_similarity_ratio(content1, content2)
    result_comparison = compare_contents(content1, content2)

    output = f"Similitud (texto llano): {similarity:.2f}%\n"
    output += "Diferencias:\n" + result_comparison + "\n"
    return output

#Función principal del difflib
def comparator_difflib(file_a, file_b):
    result_preprocessed = compare_preprocessed(file_a, file_b)
    result_plain        = compare_plain(file_a, file_b)
    return result_preprocessed, result_plain