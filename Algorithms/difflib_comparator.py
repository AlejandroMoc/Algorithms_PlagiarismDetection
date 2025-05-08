#Comparador de código mediante liffdib
## A01736339 - Jacqueline Villa Asencio
## A01736671 - José Juan Irene Cervantes
## A01736346 - Augusto Gómez Maxil
## A01736353 - Alejandro Daniel Moctezuma Cruz

import difflib, os, glob, itertools, tokenize
from io import BytesIO

## LECTURA DE ARCHIVOS
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()

def read_file_raw(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

## DIFERENCIAS TEXTO
def compare_contents(content1, content2):
    differences = difflib.ndiff(content1, content2)
    return '\n'.join(differences)

## SIMILITUD CON .ratio()
def compare_similarity_ratio(tokens1, tokens2):
    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    return matcher.ratio() * 100

## PREPROCESAMIENTO LÉXICO
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

## COMPARADOR PREPROCESADO
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

## COMPARADOR TEXTO PLANO
def compare_plain(code1, code2):
    content1 = read_file(code1)
    content2 = read_file(code2)

    similarity = compare_similarity_ratio(content1, content2)
    result_comparison = compare_contents(content1, content2)

    output = f"Similitud (texto llano): {similarity:.2f}%\n"
    output += "Diferencias:\n" + result_comparison + "\n"
    return output

## PROGRAMA PRINCIPAL
def main():
    print("== Comparador de archivos de Python usando difflib ==")

    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, 'Data', 'Luciano')
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    
    total_subcarpetas = len(subfolders)
    print(f"Se encontraron {total_subcarpetas} subcarpetas para analizar.\n")

    with open("resultado_comparador_difflib.txt", "w", encoding="utf-8") as report:
        for subfolder in subfolders:
            files = sorted(glob.glob(os.path.join(subfolder, '*.py')))
            nombre_subcarpeta = os.path.basename(subfolder)

            if len(files) < 2:
                print(f"⚠️ Subcarpeta '{nombre_subcarpeta}' ignorada (menos de 2 archivos)")
                continue

            print(f"\n🔍 Comparando archivos en subcarpeta: {nombre_subcarpeta}")

            for file1, file2 in itertools.combinations(files, 2):
                name1 = os.path.basename(file1)
                name2 = os.path.basename(file2)

                print(f"🟢 Comparando: {name1} vs {name2}")

                result_preprocessed = compare_preprocessed(file1, file2)
                result_plain = compare_plain(file1, file2)

                print("Resultado Preprocesado:")
                print(result_preprocessed)
                print("Resultado Texto Plano:")
                print(result_plain)
                print("=" * 60)

                report.write(f"\n📁 Subcarpeta: {nombre_subcarpeta}\n")
                report.write(f"Comparación de {name1} vs {name2}\n")
                report.write("Resultado Preprocesado:\n" + result_preprocessed + "\n")
                report.write("Resultado Texto Plano:\n" + result_plain + "\n")
                report.write("=" * 60 + "\n")


if __name__ == '__main__':
    main()
