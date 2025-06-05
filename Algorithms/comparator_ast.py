import io, os, ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#Leer archivo como bits
def read_file_raw(file_path: str) -> io.BufferedReader:
    with open(file_path, 'rb') as file:
        return file.read()

# Extraer features estructurales del AST
def extract_ast_features(tree) -> tuple[dict[str, int], np.ndarray, list[str]]:
    features = {
        'num_nodes': 0,
        'num_functions': 0,
        'num_loops': 0,
        'num_if': 0,
        'max_depth': 0,
        'num_variables': 0,
        'num_lists': 0,
        'num_dicts': 0
    }
    variable_names = set()

    def traverse(node, depth):
        if not isinstance(node, ast.AST):
            return
        features['num_nodes'] += 1
        features['max_depth'] = max(features['max_depth'], depth)

        if isinstance(node, ast.FunctionDef):
            features['num_functions'] += 1
        elif isinstance(node, (ast.For, ast.While)):
            features['num_loops'] += 1
        elif isinstance(node, ast.If):
            features['num_if'] += 1
        elif isinstance(node, ast.List):
            features['num_lists'] += 1
        elif isinstance(node, ast.Dict):
            features['num_dicts'] += 1
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variable_names.add(target.id)
        elif isinstance(node, ast.Name):
            variable_names.add(node.id)

        for child in ast.iter_child_nodes(node):
            traverse(child, depth + 1)

    traverse(tree, 1)
    features['num_variables'] = len(variable_names)

    return np.array(list(features.values()), dtype=float), features, list(variable_names)

# Normalizar los features a [0,1]
def normalize_features(f1, f2):
    max_vals = np.maximum(f1, f2)
    max_vals[max_vals == 0] = 1
    return f1 / max_vals, f2 / max_vals

# Veredicto de plagio
def decide_plagiarism(ted_sim, feature_sim, alpha = 0.5, threshold = 0.70):
    score = alpha * ted_sim + (1 - alpha) * feature_sim
    plagio = score >= threshold

    #Escribir el umbral en un archivo txt (depuración)
    #with open(f'threshold_report_{threshold}.txt', 'a') as file:
        #file.write(f"{score:.3f}\n")

    #print(f"✅ Score combinado: {score:.3f} (TED={ted_sim:.3f}, Features={feature_sim:.3f})")
    #print(f"📌 Umbral de decisión: {threshold}")
    #print("🛑 AST:", "PLAGIO detectado")
    
    return plagio, round(score, 3)

# Calcular similitud TED basado en la comparación de nodos
def count_common_nodes(node1, node2):
    if not isinstance(node1, ast.AST) or not isinstance(node2, ast.AST):
        return 0
    common_count = 0
    if type(node1) == type(node2):
        common_count += 1
    for child1, child2 in zip(ast.iter_child_nodes(node1), ast.iter_child_nodes(node2)):
        common_count += count_common_nodes(child1, child2)
    return common_count

# Comparar dos archivos .py
def comparator_ast(code1: str, code2: str):

    tree_ast1: ast.Module = ast.parse(code1)
    tree_ast2: ast.Module = ast.parse(code2)

    # Obtener nodos comuness, máximos y similitud TED
    common_nodes: int = count_common_nodes(tree_ast1, tree_ast2)
    max_nodes: int = max(len(list(ast.walk(tree_ast1))), len(list(ast.walk(tree_ast2))))
    ted_similarity: float = common_nodes / max_nodes if max_nodes > 0 else 0

    common_nodes: int = count_common_nodes(tree_ast1, tree_ast2)
    max_nodes: int = max(len(list(ast.walk(tree_ast1))), len(list(ast.walk(tree_ast2))))
    ted_similarity: float = common_nodes / max_nodes if max_nodes > 0 else 0

    #Obtener vectores y características
    f1_vector, f1_dict, vars1 = extract_ast_features(tree_ast1)
    f2_vector, f2_dict, vars2 = extract_ast_features(tree_ast2)
    f1_norm, f2_norm = normalize_features(f1_vector, f2_vector)
    features_sim = cosine_similarity([f1_norm], [f2_norm])[0][0]

    # Función para obtener el orden de definiciones
    def function_order(tree):
        return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]

    orden_funcs1 = function_order(tree_ast1)
    orden_funcs2 = function_order(tree_ast2)

    ast_flag_0, score = decide_plagiarism(ted_sim = ted_similarity, feature_sim = features_sim, threshold=0.9)
    ast_flag_1, score = decide_plagiarism(ted_sim = ted_similarity, feature_sim = features_sim, threshold=0.35)

    return [
        round(ted_similarity, 3),        # Índice 0: similitud TED
        round(features_sim, 3),          # Índice 1: similitud de features
        list(f1_dict.values()),          # Índice 2: features archivo 1
        list(f2_dict.values()),          # Índice 3: features archivo 2
        score,                           # Índice 4: score combinado
        vars1,                           # Índice 5: variables archivo 1
        vars2,                           # Índice 6: variables archivo 2
        orden_funcs1,                    # Índice 7: orden de funciones archivo 1
        orden_funcs2,                    # Índice 8: orden de funciones archivo 2

        #Regresar qué tipo de plagio considera
        ast_flag_0,                      # Índice 9: Plagio de tipo exacto
        ast_flag_1,                      # Índice 10: Plagio de tipo 2?
    ]

#Ejecución principal
def main():
    file_a = "Data_Check/file1.py"
    file_b = "Data_Check/file2.py"

    #Abrir archivos como bits
    content_a = read_file_raw(file_a)
    content_b = read_file_raw(file_b)

    resultado = comparator_ast(content_a, content_b)
    print(resultado)

if __name__ == '__main__':
    main()