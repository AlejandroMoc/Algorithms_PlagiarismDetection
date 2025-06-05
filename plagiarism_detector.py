import os
import glob
import numpy as np
import pandas as pd
from Algorithms.comparator_sa import comparator_sa
from Algorithms.comparator_difflib import comparator_difflib
from Algorithms.comparator_ast import comparator_ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from joblib import dump, load
from tqdm import tqdm
import difflib
import ast
import time


##FUNCIONES
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
            #Plagio tipo 0
            difflib_results = comparator_difflib(file_a, file_b)
            similarity_preprocessed, result_preprocessed, similarity_plain, result_plain = difflib_results

            #Plagio tipo 1
            sa_similarity = comparator_sa(file_a, file_b)

            #Plagio tipo 2 y 3
            result_ast = comparator_ast(file_a, file_b)
            ted_similarity = result_ast[0]  #Similitud TED
            features_similarity = result_ast[1]  #Similitud de features
            is_ast_plagiarism_0 = result_ast[9]  #Plagio tipo exacto
            is_ast_plagiarism_1 = result_ast[10]  #Plagio tipo 2

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
    model_path = 'plagiarism.joblib'
    mlb_path = 'multilabelbinarizer.joblib'

    # Si existe, cargar el modelo
    if os.path.exists(model_path) and os.path.exists(mlb_path):
        prediction_model = load(model_path)
        mlb = load(mlb_path)
        print("✅ Modelo y binarizador cargados.")

    # Si no existe, entrenar el modelo
    else:
        base_path = os.path.dirname(__file__)
        data_dir = os.path.join(base_path, 'Data')

        # ⏱️ Medición de tiempo de preprocesamiento
        print("⚙️  Generando datos de entrenamiento...")
        start_preproc = time.time()
        all_data = generate_training_data_from_leaf_dirs(data_dir)
        end_preproc = time.time()
        print(f"⏱️ Tiempo de preprocesamiento: {end_preproc - start_preproc:.2f} segundos")

        if not all_data:
            print("❌ No hay datos para entrenar.")
            return

        df = pd.DataFrame(all_data, columns=['sa_similarity', 'ted_similarity', 'features_similarity', 'is_ast_plagiarism_0', 'is_ast_plagiarism_1', 'edit_distance', 'num_nodes_diff', 'num_funcs_diff', 'num_loops_diff', 'plagiarism_type'])
        X = df[['sa_similarity', 'ted_similarity', 'edit_distance', 'num_nodes_diff', 'num_funcs_diff', 'num_loops_diff']]
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

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)

        # ⏱️ Medición de tiempo de entrenamiento
        print("🤖 Entrenando modelo...")
        start_train = time.time()
        prediction_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        prediction_model.fit(X_train, y_train)
        end_train = time.time()
        print(f"⏱️ Tiempo de entrenamiento: {end_train - start_train:.2f} segundos")

        # ⏱️ Medición de tiempo de evaluación
        print("📈 Evaluación del modelo:")
        start_eval = time.time()
        y_pred_bin = prediction_model.predict(X_test)
        print(classification_report(y_test, y_pred_bin, target_names=[f"Tipo {cls}" for cls in mlb.classes_], zero_division=0))
        end_eval = time.time()
        print(f"⏱️ Tiempo de evaluación (prueba con 30%): {end_eval - start_eval:.2f} segundos")

        # Guardar modelo y binarizador
        dump(prediction_model, model_path)
        dump(mlb, mlb_path)
        print("💾 Modelo y binarizador guardados.")

    # Predicción de prueba
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
        X_test = np.array([[sa, ted, edit_distance, num_nodes_diff, num_funcs_diff, num_loops_diff]])
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
