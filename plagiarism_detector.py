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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

##FUNCIONES

#Algoritmos
def compare_files(file_a: str, file_b: str):
    if file_a == file_b:
        return None
    elif (file_a is None) or (file_b is None):
        return None
    else:
        ##Pasar por algoritmos de preprocesamiento
        try:
            sa_similarity = comparator_sa(file_a, file_b)  # Plagio tipo 1
            result_ast = comparator_ast(file_a, file_b)    # Plagio tipo 2 y 3
            ted_similarity = result_ast[0]                 # Tree Edit Distance
            
            # Determinar el tipo de plagio
            plagio_type = determine_plagiarism_type(sa_similarity, ted_similarity)

            return [sa_similarity, ted_similarity, plagio_type]
        except SyntaxError as e:
            print(f"SyntaxError in files {file_a} and {file_b}: {e}")
            return None
        except Exception as e:
            print(f"Error comparing files {file_a} and {file_b}: {e}")
            return None

def determine_plagiarism_type(sa_similarity, ted_similarity):
    if sa_similarity > 0.8:
        return 1  #Plagio tipo 1
    elif ted_similarity > 0.5:
        return 2  #Plagio tipo 2
    else:
        return 0  #No plagio

#Ejecución principal
def main():
    print("Detector de Plagio utilizando Machine Learning")

    #Abrir BDD de Entrenamiento
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, 'Data')

    #Obtener todas las subcarpetas dentro de data_dir
    subdirs = []
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            subdirs.append(os.path.join(root, d))

    all_data = []

    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        files_training = sorted(glob.glob(os.path.join(subdir_path, '*.py')))

        #Verificar longitud mayor a 2 y que no haya carpetas
        if len(files_training) < 2 and not any(os.path.isdir(os.path.join(subdir_path, d)) for d in os.listdir(subdir_path)):
            print(f"No se encontraron suficientes archivos en {subdir}.")
            continue
        
        #print(f"Se encontraron {len(files_training)} archivos para entrenamiento en la carpeta {subdir}.\n")
        for file_a in files_training:
            for file_b in files_training:
                if file_a != file_b:
                    current_result = compare_files(file_a, file_b)
                    if current_result is not None:
                        all_data.append(current_result)

    #Convertir a DataFrame
    df = pd.DataFrame(all_data, columns=['sa_similarity', 'ted_similarity', 'plagiarism_type'])

    #Dividir datos
    X = df[['sa_similarity', 'ted_similarity']]
    y = df['plagiarism_type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Entrenar un modelo
    model = LogisticRegression()
    #Alternativa: model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    #Evaluar el modelo
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()