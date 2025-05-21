import ast
import os
import numpy as np
from zss import simple_distance, Node
from sklearn.metrics.pairwise import cosine_similarity

# Convertir AST a zss-compatible tree
def ast_to_zss(node):
    if not isinstance(node, ast.AST):
        return None
    label = node.__class__.__name__
    children = [ast_to_zss(child) for child in ast.iter_child_nodes(node)]
    return Node(label, [child for child in children if child is not None])

# Extraer features estrucutrales del AST
def extract_ast_features(tree):
    features = {
        'num_nodes' : 0,
        'num_functions' : 0,
        'num_loops' : 0,
        'num_if' : 0,
        'max_depth' : 0
    }
    
    def traverse(node, depth):
        if not isinstance(node, ast.AST):
            return 0
        
        features['num_nodes'] += 1
        features['max_depth'] = max(features['max_depth'], depth)
        
        if isinstance(node, ast.FunctionDef):
            features['num_functions'] += 1
        elif isinstance(node, (ast.For, ast.While)):
            features['num_loops'] += 1
        elif isinstance(node, ast.If):
            features['num_if'] += 1
        
        for child in ast.iter_child_nodes(node):
            traverse(child, depth + 1)
            
    traverse(tree, 1)
    
    return np.array(list(features.values()), dtype=float), features


# Normalizar los features a [0,1]
def normalize_features(f1, f2):
    max_vals = np.maximum(f1, f2)
    max_vals[max_vals == 0] = 1
    
    return f1 / max_vals, f2 /max_vals

# Leer el código fuente de Python desde el archivo
def read_code(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def decidir_plagio(ted_sim, feature_sim, alpha=0.5, umbral=0.70):
    """
    Decide si dos códigos son plagio combinando similitud TED y de features.

    Parámetros:
    - ted_sim: similitud por Tree Edit Distance (valor entre 0 y 1)
    - feature_sim: similitud estructural por features (valor entre 0 y 1)
    - alpha: peso de TED (entre 0 y 1); 1-alpha es el peso de features
    - umbral: valor mínimo para considerar plagio (entre 0 y 1)

    Retorna:
    - True si se considera plagio, False en caso contrario
    - El score combinado
    """

    score = alpha * ted_sim + (1 - alpha) * feature_sim
    plagio = score >= umbral

    print(f"✅ Score combinado: {score:.3f} (TED={ted_sim:.3f}, Features={feature_sim:.3f})")
    print(f"📌 Umbral de decisión: {umbral}")
    print("🛑 Veredicto final:", "PLAGIO" if plagio else "NO PLAGIO")

    return plagio, round(score, 3)

# Función principal
def compare_files_ast(file1, file2):
    code1 = read_code(file1)
    code2 = read_code(file2)
    
    tree_ast1 = ast.parse(code1)
    tree_ast2 = ast.parse(code2)
    
    # Calcular TED
    tree_zss1 = ast_to_zss(tree_ast1)
    tree_zss2 = ast_to_zss(tree_ast2)
    
    ted = simple_distance(tree_zss1, tree_zss2)
    
    max_nodes = max(len(list(ast.walk(tree_ast1))), len(list(ast.walk(tree_ast2))))
    
    ted_similarity = 1 - (ted / max_nodes)
    
    # Similitud de características
    f1_vector, f1_dict = extract_ast_features(tree_ast1)
    f2_vector, f2_dict = extract_ast_features(tree_ast2)
    
    f1_norm, f2_norm = normalize_features(f1_vector, f2_vector)
    
    features_sim = cosine_similarity([f1_norm], [f2_norm])[0][0]
    
    plagio, score = decidir_plagio(ted_similarity, features_sim)
    
    return {
        'ted_similarity' : round(ted_similarity, 3),
        'feature_similarity' : round(features_sim, 3),
        'features_file1' : f1_dict,
        'features_file2' : f2_dict
    }
    
# Ruta de códigos
file_a = "Data/testcase1.py"
file_b = "Data/testcase7.py"

print(compare_files_ast(file_a, file_b))