#Detector de plagio usando Python
##A01736339 - Jacqueline Villa Asencio
##A01736671 - José Juan Irene Cervantes
##A01736346 - Augusto Gómez Maxil
##A01736353 - Alejandro Daniel Moctezuma Cruz

##LIBRERÍAS
import os, glob
import numpy as np
import pandas as pd
from Algorithms.comparator_sa import comparator_sa
from Algorithms.comparator_difflib import comparator_difflib
from Algorithms.comparator_ast import comparator_ast
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from joblib import dump, load  #Importar joblib

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

            #Determinar el tipo de plagio
            plagio_type = determine_plagiarism_type(sa_similarity, ted_similarity)

            return [sa_similarity, ted_similarity, plagio_type]
        except SyntaxError as e:
            print(f"SyntaxError in files {file_a} and {file_b}: {e}")
            return None
        except Exception as e:
            print(f"Error comparing files {file_a} and {file_b}: {e}")
            return None

def determine_plagiarism_type(sa_similarity, ted_similarity):
    if sa_similarity > 0.6:  #Ajustar umbral
        return 1  #Plagio tipo 1
    elif ted_similarity > 0.4:  #Ajustar umbral
        return 2  #Plagio tipo 2
    else:
        return 0  #No plagio

#Ejecución principal
def main():
    print("Detector de Plagio utilizando Machine Learning")

    all_data = []
    model_path = 'plagiarism_detector_model.joblib'

    #Intentar cargar el modelo si existe
    if os.path.exists(model_path):
        model = load(model_path)
        print("Modelo cargado desde el archivo.")
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
                            all_data.append(current_result)

        #Convertir a DataFrame
        df = pd.DataFrame(all_data, columns=['sa_similarity', 'ted_similarity', 'plagiarism_type'])

        #Dividir los datos
        X = df[['sa_similarity', 'ted_similarity']]
        y = df['plagiarism_type']

        #Entrenar un modelo
        model = DecisionTreeClassifier()
        model.fit(X, y)

        #Guardar el modelo entrenado
        dump(model, model_path)
        print("Modelo guardado en el archivo.")

    #Archivos específicos para evaluación
    file_a = '/home/daniel/Descargas/Algorithms_PlagiarismDetection/Data/algorithms/adjacency_list/adjacency_list.py'
    file_b = '/home/daniel/Descargas/Algorithms_PlagiarismDetection/Data/algorithms/adjacency_list/adjacency_list_tipo2.py'

    #Comparar los archivos específicos
    specific_result = compare_files(file_a, file_b)
    if specific_result is not None:
        X_test = np.array([specific_result[:-1]])  #Excluir el tipo de plagio para la predicción
        y_test = np.array([specific_result[-1]])    #Tipo de plagio para evaluación

        #Predecir usando el modelo
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()