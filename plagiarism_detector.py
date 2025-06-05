##LIBRERÍAS
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, ast, glob, difflib
from joblib import dump, load
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from Algorithms.comparator_sa import comparator_sa
from Algorithms.comparator_ast import comparator_ast
from Algorithms.comparator_difflib import comparator_difflib

##FUNCIONES
#Limpieza de comentarios y espacios innecesarios
def clean_code(code):
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith('#'):
            cleaned_lines.append(line)  # Conserva la indentación original
    return '\n'.join(cleaned_lines)

#Renombramiento de variables
class VariableRenamer(ast.NodeTransformer):
    def __init__(self):
        self.variable_count = 0
        self.renamed_vars = {}

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            if node.id not in self.renamed_vars:
                new_name = f"var_{self.variable_count}"
                self.renamed_vars[node.id] = new_name
                self.variable_count += 1
            return ast.copy_location(ast.Name(id=self.renamed_vars[node.id], ctx=node.ctx), node)
        elif isinstance(node.ctx, ast.Load) and node.id in self.renamed_vars:
            return ast.copy_location(ast.Name(id=self.renamed_vars[node.id], ctx=node.ctx), node)
        return node

def rename_variables(code):
    try:
        tree = ast.parse(code)
        renamer = VariableRenamer()
        renamed_tree = renamer.visit(tree)
        ast.fix_missing_locations(renamed_tree)
        return ast.unparse(renamed_tree)
    except Exception as e:
        print(f"Error al renombrar variables: {e}")
        return code  # Devuelve el código original si falla

#Normalización del código
def normalize_code(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    cleaned_code = clean_code(code)
    renamed_code = rename_variables(cleaned_code)
    return renamed_code

#Funciones adicionales para contar nodos, funciones y bucles en el AST
def ast_nodes(file):
    with open(file, "r") as source:
        tree = ast.parse(source.read())
    return [node for node in ast.walk(tree)]

def count_functions(file):
    with open(file, "r") as source:
        tree = ast.parse(source.read())
    return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])

def count_loops(file):
    with open(file, "r") as source:
        tree = ast.parse(source.read())
    return len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))])

#Comparar archivos para entrenamiento
def compare_files(file_a: str, file_b: str):
    if file_a == file_b:
        return None
    elif (file_a is None) or (file_b is None):
        return None
    else:
        try:
            #Normalizar código
            normalized_code_a = normalize_code(file_a)
            normalized_code_b = normalize_code(file_b)

            #Plagio tipo 0
            difflib_results = comparator_difflib(normalized_code_a, normalized_code_b)
            similarity_preprocessed, result_preprocessed, similarity_plain, result_plain = difflib_results

            #Plagio tipo 1
            sa_similarity = comparator_sa(normalized_code_a, normalized_code_b)

            #Plagio tipo 2 y 3
            result_ast = comparator_ast(normalized_code_a, normalized_code_b)
            ted_similarity = result_ast[0]          #Similitud TED
            features_similarity = result_ast[1]     #Similitud de features
            is_ast_plagiarism_0 = result_ast[9]     #Plagio tipo 1 (exacto)
            is_ast_plagiarism_1 = result_ast[10]    #Plagio tipo 2

            #Nueva comparación de distancia de edición
            with open(file_a) as f1, open(file_b) as f2:
                f1_lines = f1.readlines()
                f2_lines = f2.readlines()
                d = difflib.ndiff(f1_lines, f2_lines)
                edit_distance = sum(1 for _ in d if _[0] != ' ')

            #Nuevas métricas estructurales
            num_nodes_diff = abs(len(ast_nodes(file_a)) - len(ast_nodes(file_b)))
            num_funcs_diff = abs(count_functions(file_a) - count_functions(file_b))
            num_loops_diff = abs(count_loops(file_a) - count_loops(file_b))

            #Imprimir similitudes
            print(f"Comparando {file_a} y {file_b}:")
            print(f"  - SA Similarity: {sa_similarity}, TED Similarity: {ted_similarity}, Edit Distance: {edit_distance}, AST considera plagio: {is_ast_plagiarism_0}")

            return [
                sa_similarity,
                ted_similarity,
                features_similarity,
                is_ast_plagiarism_0,
                is_ast_plagiarism_1,
                edit_distance,
                num_nodes_diff,
                num_funcs_diff,
                num_loops_diff
            ]

        except SyntaxError as e:
            print(f"Error de sintaxis en los archivos {file_a} y {file_b}: {e}")
            return None
        except Exception as e:
            print(f"Error al comparar los archivos {file_a} y {file_b}: {e}")
            return None

def generate_training_data_from_leaf_dirs(data_dir):
    data = []
    leaf_dirs = [os.path.join(root) for root, dirs, files in os.walk(data_dir)
                 if any(f.endswith(".py") for f in files)]

    print(f"🔍 Encontradas {len(leaf_dirs)} carpetas hoja.")

    for ruta_subcarpeta in tqdm(leaf_dirs, desc="Procesando carpetas hoja"):
        archivos = sorted(glob.glob(os.path.join(ruta_subcarpeta, '*.py')))
        if os.path.exists(os.path.join(ruta_subcarpeta, "plagio_tipo1.txt")):
            for f in archivos:
                result = compare_files(f, f)
                if result:
                    sa, ted, features_similarity, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff = result
                    data.append([sa, ted, features_similarity, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff, [1]])

        for i in range(len(archivos)):
            for j in range(i + 1, len(archivos)):
                file_a, file_b = archivos[i], archivos[j]
                result = compare_files(file_a, file_b)
                if result:
                    sa, ted, features_similarity, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff = result
                    name_b = os.path.basename(file_b).lower()
                    types = []
                    if "tipo0" in name_b: types.append(0)
                    if "tipo1" in name_b: types.append(1)
                    if "tipo2" in name_b: types.append(2)
                    if "tipo3" in name_b: types.append(3)
                    if not types: types.append(0)
                    data.append([sa, ted, features_similarity, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff, types])
    return data

def algorithm(test_file_a, test_file_b):
    print("🚀 Detector de Plagio utilizando Machine Learning")
    model_path = 'plagiarism_model.joblib'
    mlb_path = 'mlb_model.joblib'

    #Si existe, cargar el modelo
    if os.path.exists(model_path) and os.path.exists(mlb_path):
        prediction_model = load(model_path)
        mlb = load(mlb_path)
        print("✅ Modelo y binarizador cargados.")

    #Si no existe, entrenar el modelo
    else:
        #Abrir BDD de Entrenamiento
        base_path = os.path.dirname(__file__)
        data_dir = os.path.join(base_path, 'Data')
        print("⚙️  Generando datos de entrenamiento...")
        all_data = generate_training_data_from_leaf_dirs(data_dir)

        if not all_data:
            print("❌ No hay datos para entrenar.")
            return

        df = pd.DataFrame(all_data, columns=[
            'sa_similarity',
            'ted_similarity',
            'features_similarity',
            'is_ast_plagiarism_0',
            'is_ast_plagiarism_1',
            'edit_distance',
            'num_nodes_diff',
            'num_funcs_diff',
            'num_loops_diff',
            'plagiarism_type'
        ])

        # ✅ Usamos ahora 9 características completas
        X = df[
            ['sa_similarity', 'ted_similarity', 'features_similarity',
             'is_ast_plagiarism_0', 'is_ast_plagiarism_1',
             'edit_distance', 'num_nodes_diff', 'num_funcs_diff', 'num_loops_diff']
        ]
        y = df['plagiarism_type']

        all_labels = sum(y, [])
        total = len(all_labels)
        print("📊 Distribución de clases en el dataset:")
        for tipo in range(4):
            count = all_labels.count(tipo)
            pct = (count / total) * 100 if total else 0
            print(f"  Tipo {tipo}: {pct:.2f}% ({count} muestras)")

        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y)

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)
        print("🤖 Entrenando modelo...")
        prediction_model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced')
        prediction_model.fit(X_train, y_train)

        # Evaluar el modelo
        print("📈 Evaluación del modelo:")
        y_pred_bin = prediction_model.predict(X_test)
        print(classification_report(y_test, y_pred_bin,
              target_names=[f"Tipo {cls}" for cls in mlb.classes_], zero_division=0))

        dump(prediction_model, model_path)
        dump(mlb, mlb_path)
        print("💾 Modelo y binarizador guardados.")


    print("🔍 Ejecutando predicción de prueba...")
    result = predict_plagiarism(test_file_a, test_file_b, prediction_model, mlb)
    if result:
        predicted_types, sa, ted, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff = result
        print(f"📋 Predicción: {predicted_types}")
        print(f"  SA Similarity: {sa:.2f}")
        print(f"  TED Similarity: {ted:.2f}")
        print(f"  Edit Distance: {edit_distance}")
        print(f"  AST - Plagio exacto: {'Sí' if ast_flag_0 else 'No'}")
        print(f"  AST - Plagio parcial: {'Sí' if ast_flag_1 else 'No'}")
        print(f"  Diferencia en número de nodos: {num_nodes_diff}")
        print(f"  Diferencia en número de funciones: {num_funcs_diff}")
        print(f"  Diferencia en número de bucles: {num_loops_diff}")
    else:
        print("⚠️  No se pudo realizar la predicción.")

def predict_plagiarism(file1, file2, prediction_model, mlb):
    result = compare_files(file1, file2)
    if result:
        sa, ted, features_similarity, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff = result
        X_test = np.array([[sa, ted, features_similarity, ast_flag_0, ast_flag_1,
                            edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff]])
        y_pred_bin = prediction_model.predict(X_test)
        if y_pred_bin.ndim == 1:
            y_pred_bin = y_pred_bin.reshape(1, -1)
        predicted_types = mlb.inverse_transform(y_pred_bin)
        return predicted_types, sa, ted, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff
    return None

def main():
    print("Detector de Plagio utilizando Machine Learning")

    #Ruta de los archivos a evaluar
    test_file_a = os.path.join('Data_Check', 'file1.py')
    test_file_b = os.path.join('Data_Check', 'file2.py')

    algorithm(test_file_a, test_file_b)

#Ejecución principal
if __name__ == '__main__':
    main()
