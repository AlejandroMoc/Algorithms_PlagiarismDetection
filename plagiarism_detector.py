#Detector de plagio usando Python
## A01736339 - Jacqueline Villa Asencio
## A01736671 - José Juan Irene Cervantes
## A01736346 - Augusto Gómez Maxil
## A01736353 - Alejandro Daniel Moctezuma Cruz

##LIBRERÍAS
import os, glob

##FUNCIONES

#Algoritmos


#Ejecución principal
def main():
    print("Detector de Plagio utilizando Machine Learning")

    #Abrir BDD de Entrenamiento
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, 'Data', 'Training')
    archivos_entrenamiento = sorted(glob.glob(os.path.join(data_dir, '*.py')))
    print(f"Se encontraron {len(archivos_entrenamiento)} archivos para entrenamiento.\n")

    #preprocesar bdd
        #reemplazar nombres de vars
        #obtener palabras reservadas y cuantas veces aparecen?


    #entrenar modelo con bdd

    #Abrir BDD de Clasificación
    data_dir = os.path.join(base_path, 'Data', 'Classification')
    archivos_clasificación = sorted(glob.glob(os.path.join(data_dir, '*.py')))
    print(f"Se encontraron {len(archivos_clasificación)} archivos para entrenamiento.\n")

    #Pasar BDD_C por algoritmos

if __name__ == '__main__':
    main()