#Detector de plagio usando Python
##A01736339 - Jacqueline Villa Asencio
##A01736671 - José Juan Irene Cervantes
##A01736346 - Augusto Gómez Maxil
##A01736353 - Alejandro Daniel Moctezuma Cruz

##LIBRERÍAS
import os, glob
from Algorithms.comparator_sa import comparator_sa
from Algorithms.comparator_difflib import comparator_difflib
from Algorithms.comparator_ast import comparator_ast

##FUNCIONES

#Algoritmos
def compare_files(file_a: str, file_b: str):
    if file_a == file_b:
        return None
    elif (file_a is None) or (file_b is None):
        return None
    else:
        ##Pasar por algoritmos de preprocesamiento
        #print(f"Comparando ", os.path.basename(file_a), " y ", os.path.basename(file_b))
        sa_similarity: float                    = comparator_sa(file_a, file_b)       #plagio tipo 1
        (difflib_preprocessed, difflib_plain)   = comparator_difflib(file_a, file_b)  #plagio tipo x
        result_ast                              = comparator_ast(file_a, file_b)      #plagio tipo 2 y 3

        #porcentaje que sea tipo a,b,c?

        ##Medidas
        #Longest Common Subsequence (LCS) Ratio
        #Tree Edit Distance (TED)
        #Similitud Combinada (Weighted Similarity Score)

        #Métricas comunes
        #Precision
        #Recall
        #F1-Score
        #Accuracy

        #Matriz de similitud

        #TODO aquí debería regresar los resultados
        return ""

#Ejecución principal
def main():
    print("Detector de Plagio utilizando Machine Learning")

    #Abrir BDD de Entrenamiento
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, 'Data')

    #Obtener todas las subcarpetas dentro de data_dir
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs:
        return
    
    #Iterar sobre subcarpetas
    for subdir in subdirs:
        #Crear ruta a subcarpeta y obtener archivos
        subdir_path = os.path.join(data_dir, subdir)
        files_training = sorted(glob.glob(os.path.join(subdir_path, '*.py')))

        #Verificar longitud de los archivos
        if len(files_training) < 2:
            print(f"No se encontraron suficientes archivos en {subdir}.")
            print(f"{subdir_path} y {files_training}\n")
            continue
        
        print(f"Se encontraron {len(files_training)} archivos para entrenamiento en la carpeta {subdir}.\n")

        #PASO 1
        #Obtener comparasiones
        all_comparisons = []
        for file_a in files_training:
            for file_b in files_training:
                if file_a != file_b:  #Evitar comparar el mismo archivo
                    current_result = compare_files(file_a, file_b)
                    if current_result is not None:
                        all_comparisons.append(current_result)

        ##Entrenar algoritmo neural

        #PASO 2
        #Pasar BDD_C por algoritmo neural para obtener resultados

        #Abrir BDD de Clasificación
        #data_dir = os.path.join(base_path, 'Data')
        #archivos_clasificación = sorted(glob.glob(os.path.join(data_dir, '*.py')))
        #print(f"Se encontraron {len(archivos_clasificación)} archivos para clasificación.\n")

        #Pasar BDD_C por algoritmos

if __name__ == '__main__':
    main()