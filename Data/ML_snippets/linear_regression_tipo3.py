import numpy as np

class ModeloRegresionLineal:
    """
    Un modelo de regresión lineal simple utilizando la ecuación normal.
    """

    def __init__(self):
        self.coeficientes = None
        self.intercepto = None

    def ajustar(self, X: np.ndarray, y: np.ndarray):
        """
        Ajusta el modelo de regresión lineal a los datos de entrenamiento.

        Parámetros:
        -----------
        X : np.ndarray
            Datos de entrenamiento, forma (n_muestras, n_características).
        y : np.ndarray
            Valores objetivo, forma (n_muestras,).
        """
        # Agrega una columna de unos a X para el término de intercepción
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calcula los parámetros óptimos utilizando la pseudoinversa para estabilidad
        theta_mejor = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        # Establece el intercepto y los coeficientes
        self.intercepto = theta_mejor[0]
        self.coeficientes = theta_mejor[1:]

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Predice los valores objetivo para los datos de prueba dados.

        Parámetros:
        -----------
        X : np.ndarray
            Datos de prueba, forma (n_muestras, n_características).
        
        Devoluciones:
        --------
        np.ndarray
            Valores objetivo predichos, forma (n_muestras,).
        """
        return self.intercepto + X.dot(self.coeficientes)


# Uso de ejemplo
if __name__ == "__main__":
    # Conjunto de datos de ejemplo
    X_entrenamiento = np.array([[1], [2], [3], [4], [5]])
    y_entrenamiento = np.array([1.5, 1.8, 3.2, 3.9, 5.1])

    # Inicializa y ajusta el modelo
    modelo = ModeloRegresionLineal()
    modelo.ajustar(X_entrenamiento, y_entrenamiento)

    # Predicciones
    X_prueba = np.array([[1.5], [3.5], [6]])
    predicciones = modelo.predecir(X_prueba)
    print("Predicciones:", predicciones)