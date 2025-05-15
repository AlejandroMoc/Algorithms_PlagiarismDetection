#Detector de plagio usando Python
## A01736339 - Jacqueline Villa Asencio
## A01736671 - José Juan Irene Cervantes
## A01736346 - Augusto Gómez Maxil
## A01736353 - Alejandro Daniel Moctezuma Cruz

##LIBRERÍAS
import os, glob
from Algorithms.sa_comparator import sa_comparator

##FUNCIONES

#Algoritmos
def compare_files(file_a: str, file_b: str):
    if file_a == file_b:
        return None
    elif (file_a is None) or (file_b is None):
        return None
    else:
        print(f"Comparando ", os.path.basename(file_a), " y ", os.path.basename(file_b))
        ##Pasar por algoritmos de preprocesamiento
        sa_similarity: float = sa_comparator(file_a, file_b)
        #Algoritmo x
        #Algoritmo x
        #Algoritmo x
        #Algoritmo x
        return ""

#Ejecución principal
def main():
    print("Detector de Plagio utilizando Machine Learning")

    # Abrir BDD de Entrenamiento
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, 'Data', 'Training')
    files_training = sorted(glob.glob(os.path.join(data_dir, '**', '*.py'), recursive = True))
    print(f"Se encontraron {len(files_training)} archivos para entrenamiento.\n")

    # PASO 1
    # Entrenar Algoritmo neural con BDD Entrenamiento
    
    ##Obtener todos los resultados
    all_comparisons = []
    for file_a in files_training:
        for file_b in files_training:
            current_result = compare_files(file_a, file_b)

            #Si el resultado existe, 
            if current_result != None:
                all_comparisons.append(current_result)

    ##Entrenar algoritmo neural

    

    # PASO 2
    # Pasar BDD_C por algoritmo neural para obtener resultados

    #Abrir BDD de Clasificación
    data_dir = os.path.join(base_path, 'Data', 'Classification')
    archivos_clasificación = sorted(glob.glob(os.path.join(data_dir, '*.py')))
    print(f"Se encontraron {len(archivos_clasificación)} archivos para clasificación.\n")

    #Pasar BDD_C por algoritmos

if __name__ == '__main__':
    main()