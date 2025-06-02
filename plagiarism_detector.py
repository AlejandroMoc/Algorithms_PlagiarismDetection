##LIBRERÍAS
import os, glob
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

##FUNCIONES
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
            ted_similarity = result_ast[0]
            fea_similarity = result_ast[1]
            # features_file1 = result_ast[2]
            # features_file2 = result_ast[3]
            # combined_score = result_ast[4]
            # variables_file1 = result_ast[5]
            # variables_file2 = result_ast[6]
            # function_order_file1 = result_ast[7]
            # function_order_file2 = result_ast[8]
            is_ast_plagiarism_0 = result_ast[9]
            is_ast_plagiarism_1 = result_ast[10]

            #Imprimir similitudes
            print(f"Comparando {file_a} y {file_b}:")
            print(f"  - SA Similarity: {sa_similarity}, TED Similarity: {ted_similarity}, Features Similarity: {fea_similarity}, Plain Similarity: {similarity_plain}, AST considers it plagiarism: {is_ast_plagiarism_0}")

            return [sa_similarity, ted_similarity, similarity_plain, is_ast_plagiarism_0, is_ast_plagiarism_1]

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
                    sa, ted, sim_plain, ast_flag_0, ast_flag_1 = result
                    data.append([sa, ted, sim_plain, ast_flag_0, ast_flag_1, [1]])

        for i in range(len(archivos)):
            for j in range(i + 1, len(archivos)):
                file_a, file_b = archivos[i], archivos[j]
                result = compare_files(file_a, file_b)
                if result:
                    sa, ted, sim_plain, ast_flag_0, ast_flag_1 = result
                    name_b = os.path.basename(file_b).lower()
                    types = []
                    if "tipo0" in name_b: types.append(0)
                    if "tipo1" in name_b: types.append(1)
                    if "tipo2" in name_b: types.append(2)
                    if "tipo3" in name_b: types.append(3)
                    if not types: types.append(0)
                    data.append([sa, ted, sim_plain, ast_flag_0, ast_flag_1, types])
    return data

def algorithm(test_file_a, test_file_b):
    print("🚀 Detector de Plagio utilizando Machine Learning")
    model_path = 'plagiarism_model.joblib'
    mlb_path = 'mlb.joblib'

    #Si existe, cargar el modelo
    if os.path.exists(model_path) and os.path.exists(mlb_path):
        model = load(model_path)
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

        df = pd.DataFrame(all_data, columns=['sa_similarity', 'ted_similarity', 'similarity_plain', 'is_ast_plagiarism_0','is_ast_plagiarism_1', 'plagiarism_type'])
        X = df[['sa_similarity', 'ted_similarity']]
        y = df['plagiarism_type']

        all_labels = sum(y, [])
        total = len(all_labels)
        print("\n📊 Distribución de clases en el dataset:")
        for tipo in range(4):
            count = all_labels.count(tipo)
            pct = (count / total) * 100 if total else 0
            print(f"  Tipo {tipo}: {pct:.2f}% ({count} muestras)")

        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y)
        
        #Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)
        print("🤖 Entrenando modelo...")
        model = RandomForestClassifier(n_estimators = 100, random_state = 42)
        model.fit(X_train, y_train)

        #Evaluar el modelo
        print("📈 Evaluación del modelo:")
        y_pred_bin = model.predict(X_test)
        print(classification_report(y_test, y_pred_bin, target_names=[f"Tipo {cls}" for cls in mlb.classes_], zero_division=0))

        dump(model, model_path)
        dump(mlb, mlb_path)
        print("💾 Modelo y binarizador guardados.")

    print("🔎 Ejecutando predicción de prueba...")
    result = predict_plagiarism(test_file_a, test_file_b, model, mlb)
    if result:
        predicted_types, sa, ted, ast_flag_0, ast_flag_1 = result
        print(f"📌 Predicción: {predicted_types}")
        print(f"  SA Similarity: {sa:.2f}")
        print(f"  TED Similarity: {ted:.2f}")
        print(f"  AST - Plagio exacto: {'Sí' if ast_flag_0 else 'No'}")
        print(f"  AST - Plagio parcial: {'Sí' if ast_flag_1 else 'No'}")
    else:
        print("⚠️  No se pudo realizar la predicción.")

def predict_plagiarism(file1, file2, model, mlb):
    result = compare_files(file1, file2)
    if result:
        sa, ted, sim_plain, ast_flag_0, ast_flag_1 = result
        X_test = np.array([[sa, ted]])
        y_pred_bin = model.predict(X_test)
        if y_pred_bin.ndim == 1:
            y_pred_bin = y_pred_bin.reshape(1, -1)
        predicted_types = mlb.inverse_transform(y_pred_bin)
        return predicted_types, sa, ted, ast_flag_0, ast_flag_1
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
