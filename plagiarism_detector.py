
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
            is_ast_plagiarism = result_ast[6]

            #Imprimir similitudes para depuración
            print(f"Comparando {file_a} y {file_b}:")
            print(f"  - SA Similarity: {sa_similarity}, TED Similarity: {ted_similarity}, Plain Similarity: {similarity_plain}, AST considers it plagiarism: {is_ast_plagiarism}")

            return [sa_similarity, ted_similarity, similarity_plain, is_ast_plagiarism]

        except SyntaxError as e:
            print(f"Error de sintaxis en los archivos {file_a} y {file_b}: {e}")
            return None
        except Exception as e:
            print(f"Error al comparar los archivos {file_a} y {file_b}: {e}")
            return None

#Determinar tipo de plagio en base a métricas y umbrales
def determine_plagiarism_type(sa_similarity, ted_similarity, similarity_plain):
    plagiarism_types = []

    #Plagio tipo 0
    if similarity_plain > 0.5:
        plagiarism_types.append(0)

    #Plagio tipo 1
    if sa_similarity > 0.6:
        plagiarism_types.append(1)

    #Plagio tipo 2
    if ted_similarity > 0.4:
        plagiarism_types.append(2)

    if plagiarism_types:
        return plagiarism_types
    else:
        return [0]

#Predecir plagio en base a modelo y archivos
def predict_plagiarism(file1, file2, model, mlb):
    comparison_result = compare_files(file1, file2)

    if comparison_result is not None:

        #Desempacar resultados de compare_files
        sa_similarity, ted_similarity, similarity_plain, is_ast_plagiarism = comparison_result

        #Predecir tipos de plagio
        X_test = np.array([[sa_similarity, ted_similarity]])
        y_pred_bin = model.predict(X_test)
        if y_pred_bin.ndim == 1:
            y_pred_bin = y_pred_bin.reshape(1, -1)
        predicted_types = mlb.inverse_transform(y_pred_bin)

        #Regresar tipos de predicción y métricas
        specific_result = predicted_types, sa_similarity, ted_similarity, is_ast_plagiarism

        return specific_result
    else:
        return None

#Algoritmo
def algorithm(test_file_a, test_file_b):
    print("Detector de Plagio utilizando Machine Learning")

    all_data = []
    model_path = 'plagiarism_detector_model.joblib'
    mlb_path = 'mlb.joblib'

    #Intentar cargar el modelo si existe
    if os.path.exists(model_path) and os.path.exists(mlb_path):
        model = load(model_path)
        mlb = load(mlb_path)
        print("Modelo y binarizador cargados desde archivos.")

    #Si no existe, entrenar el modelo
    else:
        #Abrir BDD de Entrenamiento
        base_path = os.path.dirname(__file__)
        data_dir = os.path.join(base_path, 'Data')

        #Obtener todas las subcarpetas dentro de data_dir
        subdirs = []
        for root, dirs, files in os.walk(data_dir):
            for d in dirs:
                subdirs.append(os.path.join(root, d))

        #Comparar archivos entre diferentes subcarpetas
        for i in range(len(subdirs)):
            for j in range(i + 1, len(subdirs)):
                subdir_a = subdirs[i]
                subdir_b = subdirs[j]

                files_a = sorted(glob.glob(os.path.join(subdir_a, '*.py')))
                files_b = sorted(glob.glob(os.path.join(subdir_b, '*.py')))

                for file_a in files_a:
                    for file_b in files_b:
                        current_result = compare_files(file_a, file_b)
                        if current_result is not None:

                            #Desempacar resultados de compare_files
                            sa_similarity, ted_similarity, similarity_plain, is_ast_plagiarism = current_result

                            #Determinar tipo de plagio
                            types = []
                            if "type_0" in file_b:
                                types.append(0)
                            if "type_1" in file_b:
                                types.append(1)
                            if "type_2" in file_b:
                                types.append(2)
                            if "type_3" in file_b:
                                types.append(3)

                            if not types:
                                types.append(0)  # Valor por defecto

                            all_data.append([sa_similarity, ted_similarity, types])

        #Convertir a DataFrame
        df = pd.DataFrame(all_data, columns=['sa_similarity', 'ted_similarity', 'plagiarism_type'])

        # Verificar si hay datos suficientes
        if df.empty:
            print("No se encontraron datos suficientes para entrenar el modelo.")
            return

        #Dividir los datos
        X = df[['sa_similarity', 'ted_similarity']]
        y = df['plagiarism_type']

        #Preparar binarización para multilabel
        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y)

        #Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

        #Entrenar un modelo multilabel
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        #Evaluar el modelo
        y_pred_bin = model.predict(X_test)
        print("Reporte de Clasificación:")
        print(classification_report(y_test, y_pred_bin, target_names=[f"Tipo {cls}" for cls in mlb.classes_]))

        #Guardar el modelo y el binarizador
        dump(model, model_path)
        dump(mlb, mlb_path)
        print("Modelo y binarizador guardados en archivos.")

    #Archivos específicos para evaluación
    specific_result = predict_plagiarism(test_file_a, test_file_b, model, mlb)

    if specific_result:
        #Desempacar resultados de predict_plagiarism
        predicted_types, sa_similarity, ted_similarity, is_ast_plagiarism = specific_result

        #Predicción de tipos de plagio
        print(f"Tipos de plagio predichos entre {test_file_a} y {test_file_b}: {predicted_types}")

        #Métricas relevantes (mostrar en el sitio web)
        print(f"Similitud SA: {sa_similarity}")
        print(f"Similitud TED: {ted_similarity}")
        print(f"AST considera que es plagio: {is_ast_plagiarism}")

    else:
        print("No se pudo predecir el tipo de plagio.")

def main():
    print("Detector de Plagio utilizando Machine Learning")

    #Ruta de los archivos a evaluar
    test_file_a = os.path.join('Data_Check', 'file1.py')
    test_file_b = os.path.join('Data_Check', 'file2.py')

    algorithm(test_file_a, test_file_b)

#Ejecución principal
if __name__ == '__main__':
    main()
