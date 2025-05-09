#Detector de plagio usando Python
## A01736339 - Jacqueline Villa Asencio
## A01736671 - José Juan Irene Cervantes
## A01736346 - Augusto Gómez Maxil
## A01736353 - Alejandro Daniel Moctezuma Cruz

##LIBRERÍAS
import os, glob

##FUNCIONES

#Algoritmos
def compare_files(file_a, file_b):
    if file_a == file_b:
        return
    else:
        #Comparar archivos
        print("Comparando ", file_a.__name__, " y ", file_b.__name__)
        
        #Pasar por algoritmos de preprocesamiento
        #preprocesar bdd
            #reemplazar nombres de vars
            #obtener palabras reservadas y cuantas veces aparecen?
            #por ejemplo
        
        return " "

#Ejecución principal
def main():
    print("Detector de Plagio utilizando Machine Learning")

    #Abrir BDD de Entrenamiento
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, 'Data', 'Training')
    archivos_entrenamiento = sorted(glob.glob(os.path.join(data_dir, '*.py')))
    print(f"Se encontraron {len(archivos_entrenamiento)} archivos para entrenamiento.\n")

    # PASO 1
    # Entrenar Algoritmo neural con BDD Entrenamiento
    
    ##Obtener todos los resultados
    all_comparisons = []
    for file_a in archivos_entrenamiento:
        for archivo_2 in archivos_entrenamiento:
            result = compare_files(file_a, file_b)

            #Si el resultado existe, 
            if result != None:
                all_comparisons.append(result)

    ##Entrenar algoritmo neural

    

    # PASO 2
    # Pasar BDD_C por algoritmo neural para obtener resultados

    #Abrir BDD de Clasificación
    data_dir = os.path.join(base_path, 'Data', 'Classification')
    archivos_clasificación = sorted(glob.glob(os.path.join(data_dir, '*.py')))
    print(f"Se encontraron {len(archivos_clasificación)} archivos para entrenamiento.\n")

    #Pasar BDD_C por algoritmos

if __name__ == '__main__':
    main()