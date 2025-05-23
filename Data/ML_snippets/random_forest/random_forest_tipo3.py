python
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

class BosqueAleatorio:
    """
    Clasificador de Bosque Aleatorio.

    Parameters:
    -----------
    n_arboles : int
        El número de árboles en el bosque.
    max_profundidad : int, opcional
        La máxima profundidad de los árboles individuales.
    min_muestras_division : int, opcional
        El número mínimo de muestras requeridas para dividir un nodo interno.
    bootstrap : bool, opcional
        Si se deben usar muestras de arranque para cada árbol (por defecto es True).
    """

    def __init__(self, n_arboles: int = 10, max_profundidad: int = None, min_muestras_division: int = 2, bootstrap: bool = True):
        self.n_arboles = n_arboles
        self.max_profundidad = max_profundidad
        self.min_muestras_division = min_muestras_division
        self.bootstrap = bootstrap
        self.arboles = []

    def ajustar(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta el bosque aleatorio a los datos de entrenamiento.

        Parameters:
        -----------
        X : np.ndarray
            Datos de entrenamiento, forma (n_muestras, n_características).
        y : np.ndarray
            Valores objetivo, forma (n_muestras,).
        """
        self.arboles = []
        for _ in range(self.n_arboles):
            # Generar muestra de arranque
            if self.bootstrap:
                indices = np.random.choice(len(X), len(X), replace=True)
            else:
                indices = np.arange(len(X))
            X_muestra = X[indices]
            y_muestra = y[indices]
            
            # Crear y entrenar un nuevo árbol de decisión
            arbol = DecisionTreeClassifier(max_depth=self.max_profundidad, min_samples_split=self.min_muestras_division)
            arbol.fit(X_muestra, y_muestra)
            self.arboles.append(arbol)

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Predice las etiquetas de clase para las muestras de entrada.

        Parameters:
        -----------
        X : np.ndarray
            Datos de entrada, forma (n_muestras, n_características).
        
        Returns:
        --------
        np.ndarray
            Etiquetas de clase predichas, forma (n_muestras,).
        """
        # Recoger predicciones de cada árbol
        predicciones_arbol = np.array([arbol.predict(X) for arbol in self.arboles])
        
        # Votación mayoritaria
        predicciones_finales = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=predicciones_arbol)
        return predicciones_finales


# Uso de ejemplo
if __name__ == "__main__":
    # Dataset de juguete
    X_entrenamiento = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_entrenamiento = np.array([0, 1, 0, 1])

    # Inicializar y ajustar el modelo
    modelo = BosqueAleatorio(n_arboles=5, max_profundidad=3, bootstrap=True)
    modelo.ajustar(X_entrenamiento, y_entrenamiento)

    # Predicciones
    X_prueba = np.array([[2, 3], [6, 7]])
    predicciones = modelo.predecir(X_prueba)
    print("Predicciones:", predicciones)