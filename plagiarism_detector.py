
##LIBRERÍAS
import os, glob
import numpy as np
import pandas as pd
from Algorithms.comparator_sa import comparator_sa
from Algorithms.comparator_difflib import comparator_difflib
from Algorithms.comparator_ast import comparator_ast
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from joblib import dump, load

##FUNCIONES

#Algoritmos
def compare_files(file_a: str, file_b: str):
    if file_a == file_b:
        return None
    elif (file_a is None) or (file_b is None):
        return None
    else:
        try:
            sa_similarity = comparator_sa(file_a, file_b)  #Plagio tipo 1
            result_ast = comparator_ast(file_a, file_b)    #Plagio tipo 2 y 3
            ted_similarity = result_ast[0]                 #Tree Edit Distance

            #Imprimir similitudes para depuración
            print(f"Comparando {file_a} y {file_b}:")
            print(f"  - SA Similarity: {sa_similarity}, TED Similarity: {ted_similarity}")

            return [sa_similarity, ted_similarity]
        except SyntaxError as e:
            print(f"Error de sintaxis en los archivos {file_a} y {file_b}: {e}")
            return None
        except Exception as e:
            print(f"Error al comparar los archivos {file_a} y {file_b}: {e}")
            return None

def determine_plagiarism_type(sa_similarity, ted_similarity):
    plagiarism_types = []

    if sa_similarity > 0.6:
        plagiarism_types.append(1)  #Plagio tipo 1
    if ted_similarity > 0.4:
        plagiarism_types.append(2)  #Plagio tipo 2

    if plagiarism_types:
        return plagiarism_types
    else:
        return [0]

#Ejecución principal
def main():
    print("Detector de Plagio utilizando Machine Learning")

    all_data = []
    model_path = 'plagiarism_detector_model.joblib'
    mlb_path = 'mlb.joblib'

    #Intentar cargar el modelo si existe
    if os.path.exists(model_path) and os.path.exists(mlb_path):
        model = load(model_path)
        mlb = load(mlb_path)
        print("Modelo y binarizador cargados desde archivos.")
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
                            sa_similarity, ted_similarity = current_result

                            #Determine plagiarism type based on file names or any logic
                            types = []
                            if "type_1" in file_b:
                                types.append(1)
                            if "type_2" in file_b:
                                types.append(2)
                            if not types:
                                types.append(0)

                            all_data.append([sa_similarity, ted_similarity, types])

        #Convertir a DataFrame
        df = pd.DataFrame(all_data, columns=['sa_similarity', 'ted_similarity', 'plagiarism_type'])

        #Dividir los datos
        X = df[['sa_similarity', 'ted_similarity']]
        y = df['plagiarism_type']

        #Preparar binarización para multilabel
        mlb = MultiLabelBinarizer()
        y_bin = mlb.fit_transform(y)

        #Entrenar un modelo multilabel
        model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))
        model.fit(X, y_bin)

        #Guardar el modelo y el binarizador
        dump(model, model_path)
        dump(mlb, mlb_path)
        print("Modelo y binarizador guardados en archivos.")

    #Archivos específicos para evaluación
    test_file_a = os.path.join('Data_Check', 'file1.py')
    test_file_b = os.path.join('Data_Check', 'file2.py')

    specific_result = compare_files(test_file_a, test_file_b)

    if specific_result is not None:
        sa_similarity, ted_similarity = specific_result
        X_test = np.array([[sa_similarity, ted_similarity]])

        #Predecir usando el modelo multilabel
        y_pred_bin = model.predict(X_test)
        plagiarism_types = determine_plagiarism_type(sa_similarity, ted_similarity)
        y_true_bin = mlb.transform([plagiarism_types])

        print(f"Similitud SA: {sa_similarity}")
        print(f"Similitud TED: {ted_similarity}")
        print(f"Tipo de Plagio real: {plagiarism_types}")
        print("Reporte de Clasificación:")
        print(classification_report(y_true_bin, y_pred_bin, target_names=[f"Tipo {cls}" for cls in mlb.classes_]))

if __name__ == '__main__':
    main()
