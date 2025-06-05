## LIBRERÍAS
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, ast, glob, difflib
from joblib import dump, load
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

from Algorithms.comparator_sa import comparator_sa
from Algorithms.comparator_ast import comparator_ast
from Algorithms.comparator_difflib import comparator_difflib
from sklearn.multioutput import MultiOutputClassifier
import sys
sys.stdout.reconfigure(encoding='utf-8')



## FUNCIONES AUXILIARES
def clean_code(code):
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line and not stripped_line.startswith('#'):
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

class VariableRenamer(ast.NodeTransformer):
    def _init_(self):
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
    except Exception:
        return code

def normalize_code(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    cleaned_code = clean_code(code)
    renamed_code = rename_variables(cleaned_code)
    return renamed_code

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


## COMPARACIÓN DE ARCHIVOS
def compare_files(file_a: str, file_b: str):
    try:
        normalized_code_a = normalize_code(file_a)
        normalized_code_b = normalize_code(file_b)

        difflib_results = comparator_difflib(normalized_code_a, normalized_code_b)
        similarity_preprocessed, _, similarity_plain, _ = difflib_results
        sa_similarity = comparator_sa(normalized_code_a, normalized_code_b)

        result_ast = comparator_ast(normalized_code_a, normalized_code_b)
        ted_similarity = result_ast[0]
        features_similarity = result_ast[1]
        is_ast_plagiarism_0 = result_ast[9]
        is_ast_plagiarism_1 = result_ast[10]

        with open(file_a) as f1, open(file_b) as f2:
            f1_lines = f1.readlines()
            f2_lines = f2.readlines()
            d = difflib.ndiff(f1_lines, f2_lines)
            edit_distance = sum(1 for _ in d if _[0] != ' ')

        num_nodes_diff = abs(len(ast_nodes(file_a)) - len(ast_nodes(file_b)))
        num_funcs_diff = abs(count_functions(file_a) - count_functions(file_b))
        num_loops_diff = abs(count_loops(file_a) - count_loops(file_b))

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
    except Exception:
        return None


## GENERACIÓN DE DATOS
def generate_training_data_from_leaf_dirs(data_dir):
    data = []
    leaf_dirs = [os.path.join(root) for root, _, files in os.walk(data_dir)
                 if any(f.endswith(".py") for f in files)]

    print(f"Encontradas {len(leaf_dirs)} carpetas hoja.")

    for carpeta in tqdm(leaf_dirs, desc="Procesando carpetas hoja"):
        archivos = glob.glob(os.path.join(carpeta, "*.py"))
        archivos = sorted(archivos)

        originales = [a for a in archivos if not any(sufijo in a for sufijo in ["_tipo1", "_tipo2", "_tipo3", "_tipo0"])]

        for original in originales:
            for modificado in archivos:
                if original == modificado:
                    continue

                # Comparar
                features = compare_files(original, modificado)
                if features is None:
                    continue

                nombre = os.path.basename(modificado).lower()
                tipo_1 = nombre.endswith("_tipo1.py")
                tipo_2 = nombre.endswith("_tipo2.py")
                tipo_3 = nombre.endswith("_tipo3.py")
                tipo_0 = nombre.endswith("_tipo0.py")

                is_plagiarism = int(tipo_1 or tipo_2 or tipo_3)
                
                # Solo tomar si es comparación válida
                data.append(features + [
                    is_plagiarism,
                    int(tipo_1), int(tipo_2), int(tipo_3)
                ])
    
    return data


## ENTRENAMIENTO Y EVALUACIÓN
def algorithm(test_file_a, test_file_b):
    print("Detector de Plagio (Binario)")
    model_path = 'plagiarism_model_binario.joblib'

    if os.path.exists(model_path):
        prediction_model = load(model_path)
        print("Modelo cargado.")
    else:
        base_path = os.path.dirname(__file__)
        data_dir = os.path.join(base_path, 'Data')

        print("Generando datos de entrenamiento...")
        start_preproc = time.time()
        all_data = generate_training_data_from_leaf_dirs(data_dir)
        end_preproc = time.time()
        print(f"Tiempo de preprocesamiento: {end_preproc - start_preproc:.2f} s")

        if not all_data:
            print("No se encontraron datos.")
            return

        df = pd.DataFrame(all_data, columns=[
            'sa_similarity', 'ted_similarity', 'features_similarity',
            'is_ast_plagiarism_0', 'is_ast_plagiarism_1', 'edit_distance',
            'num_nodes_diff', 'num_funcs_diff', 'num_loops_diff', 'plagiarism'
        ])

        X = df.drop(columns=['plagiarism'])
        y = df['plagiarism']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        print("Entrenando modelo...")
        start_train = time.time()
        prediction_model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced')
        prediction_model.fit(X_train, y_train)
        end_train = time.time()
        print(f"Tiempo de entrenamiento: {end_train - start_train:.2f} s")

        print("Evaluación del modelo:")
        y_pred = prediction_model.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["No Plagio", "Plagio"], zero_division=0))

        dump(prediction_model, model_path)
        print("Modelo guardado.")

    result = predict_plagiarism(test_file_a, test_file_b, prediction_model)
    if result:
        pred, sa, ted, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff = result
        print(f"Plagio detectado: {'Sí' if pred == 1 else 'No'}")
        print(f"  SA Similarity: {sa:.2f}")
        print(f"  TED Similarity: {ted:.2f}")
        print(f"  Edit Distance: {edit_distance}")
        print(f"  AST - Plagio exacto: {'Sí' if ast_flag_0 else 'No'}")
        print(f"  AST - Plagio parcial: {'Sí' if ast_flag_1 else 'No'}")
        print(f"  Nodos: {num_nodes_diff}, Funciones: {num_funcs_diff}, Bucles: {num_loops_diff}")
    else:
        print("No se pudo realizar la predicción.")
    
    entrenar_modelo_multilabel(all_data)


def predict_plagiarism(file1, file2, model):
    result = compare_files(file1, file2)
    if result:
        sa, ted, features_similarity, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff = result
        X_test = np.array([[sa, ted, features_similarity, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff]])
        pred = model.predict(X_test)[0]
        return pred, sa, ted, ast_flag_0, ast_flag_1, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff
    return None

def entrenar_modelo_multilabel(data):
    if not data:
        print("❌ No hay datos para entrenar el modelo multilabel.")
        return

    df = pd.DataFrame(data, columns=[
        'sa_similarity', 'ted_similarity', 'features_similarity',
        'is_ast_plagiarism_0', 'is_ast_plagiarism_1', 'edit_distance',
        'num_nodes_diff', 'num_funcs_diff', 'num_loops_diff',
        'plagiarism', 'tipo1', 'tipo2', 'tipo3'
    ])

    X = df.drop(columns=['plagiarism', 'tipo1', 'tipo2', 'tipo3'])
    y = df[['tipo1', 'tipo2', 'tipo3']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = MultiOutputClassifier(RandomForestClassifier(n_estimators=300, random_state=42))
    modelo.fit(X_train, y_train)

    dump(modelo, "modelo_multilabel.joblib")
    print("✅ Modelo multilabel entrenado y guardado.")

def predecir_tipos(file1, file2):
    model_path = "modelo_multilabel.joblib"
    if not os.path.exists(model_path):
        print("No se ha entrenado el modelo multilabel.")
        return []

    model = load(model_path)
    result = compare_files(file1, file2)
    if not result:
        return []

    X = np.array([result])
    pred = model.predict(X)[0]
    tipos = []
    if pred[0]: tipos.append("Tipo 1")
    if pred[1]: tipos.append("Tipo 2")
    if pred[2]: tipos.append("Tipo 3")
    return tipos


 
def compare_all_pairs():
    upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
    if not os.path.exists(upload_folder):
        print("La carpeta 'uploads' no existe.")
        return []

    archivos = [f for f in os.listdir(upload_folder) if f.endswith('.py')]
    rutas = [os.path.join(upload_folder, f) for f in archivos]

    if len(rutas) < 2:
        print("Se necesitan al menos dos archivos para comparar.")
        return []

    # ⚠️ Verificar y entrenar modelo binario si no existe
    model_path = 'plagiarism_model_binario.joblib'
    if not os.path.exists(model_path):
        print("Modelo binario no encontrado. Entrenando ahora...")
        base_path = os.path.dirname(__file__)
        data_dir = os.path.join(base_path, 'Data')
        all_data = generate_training_data_from_leaf_dirs(data_dir)
        if all_data:
            df = pd.DataFrame(all_data, columns=[
                'sa_similarity', 'ted_similarity', 'features_similarity',
                'is_ast_plagiarism_0', 'is_ast_plagiarism_1', 'edit_distance',
                'num_nodes_diff', 'num_funcs_diff', 'num_loops_diff',
                'plagiarism', 'tipo1', 'tipo2', 'tipo3'
            ])
            X = df.drop(columns=['plagiarism', 'tipo1', 'tipo2', 'tipo3'])
            y = df['plagiarism']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            modelo_binario = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced')
            modelo_binario.fit(X_train, y_train)
            dump(modelo_binario, model_path)
            print("✅ Modelo binario entrenado y guardado.")
        else:
            print("❌ No se pudo entrenar el modelo binario. Datos insuficientes.")
            return

    model = load(model_path)

    # ⚠️ Verificar y entrenar modelo multilabel si no existe
    if not os.path.exists("modelo_multilabel.joblib"):
        print("Modelo multilabel no encontrado. Entrenando ahora...")
        base_path = os.path.dirname(__file__)
        data_dir = os.path.join(base_path, 'Data')
        all_data = generate_training_data_from_leaf_dirs(data_dir)
        if all_data:
            entrenar_modelo_multilabel(all_data)
        else:
            print("No se pudo entrenar modelo multilabel. Datos insuficientes.")

    resultados = []

    for i in range(len(rutas)):
        for j in range(i + 1, len(rutas)):
            file_a = rutas[i]
            file_b = rutas[j]
            try:
                result = predict_plagiarism(file_a, file_b, model)
                if result:
                    pred, sa, ted, plagio_0, plagio_1, ed, n, f, l = result
                    nombre_a = os.path.basename(file_a)
                    nombre_b = os.path.basename(file_b)
                    tipos = predecir_tipos(file_a, file_b)

                    fila = [
                        nombre_a,
                        nombre_b,
                        "Sí" if pred == 1 else "No",
                        tipos if pred == 1 else [],
                        round(sa * 100, 2)
                    ]
                    resultados.append(fila)
                    print(f"Comparación: {fila}")
            except Exception as e:
                print(f"Error comparando {file_a} y {file_b}: {e}")

    # Imprimir tabla formateada en consola
    print("\n📊 Resultados de Comparación:\n")

    ancho_archivo = 25
    ancho_plagio = 10
    ancho_tipo = 18
    ancho_similitud = 16

    encabezado = (
        "Archivo 1".ljust(ancho_archivo) +
        "Archivo 2".ljust(ancho_archivo) +
        "¿Plagio?".ljust(ancho_plagio) +
        "Tipo(s)".ljust(ancho_tipo) +
        "Similitud SA (%)".ljust(ancho_similitud)
    )
    print(encabezado)
    print("-" * (ancho_archivo * 2 + ancho_plagio + ancho_tipo + ancho_similitud))

    for fila in resultados:
        archivo1, archivo2, plagio, tipos, similitud = fila
        tipos_str = ", ".join(tipos) if tipos else "-"
        print(
            archivo1.ljust(ancho_archivo) +
            archivo2.ljust(ancho_archivo) +
            plagio.ljust(ancho_plagio) +
            tipos_str.ljust(ancho_tipo) +
            f"{similitud:.2f}".ljust(ancho_similitud)
        )

    return resultados




def main():
    print("Detector de Plagio Binario")
    # test_file_a = os.path.join('Data_Check', 'file1.py')
    # test_file_b = os.path.join('Data_Check', 'file2.py')
    # algorithm(test_file_a, test_file_b)
    print("Comparando todos los archivos de 'uploads/'...")
    compare_all_pairs()


if __name__ == '__main__':
    main()