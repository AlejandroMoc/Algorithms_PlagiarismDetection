import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

class ModeloLogistico:
    """
    Modelo de Regresión Logística.
    """

    def __init__(self):
        """
        Inicializa el ModeloLogistico con un modelo LogisticRegression.
        """
        self.modelo = LogisticRegression()
        self.registro = logging.getLogger(__name__)
        self.registro.info("Modelo LogisticRegression inicializado.")

    def ajustar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el modelo de Regresión Logística a los datos.

        Parámetros:
        X (np.ndarray): La matriz de características.
        y (np.ndarray): El vector objetivo.
        """
        try:
            self.modelo.fit(X, y)
            self.registro.info("Ajuste del modelo exitoso.")
        except Exception as e:
            self.registro.error(f"Error durante el ajuste del modelo: {e}")
            raise

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Predice usando el modelo de Regresión Logística.

        Parámetros:
        X (np.ndarray): La matriz de características.

        Retorna:
        np.ndarray: Los valores predichos.
        """
        try:
            return self.modelo.predict(X)
        except Exception as e:
            self.registro.error(f"Error durante la predicción: {e}")
            raise

    def puntuacion(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula la puntuación del modelo de Regresión Logística.

        Parámetros:
        X (np.ndarray): La matriz de características.
        y (np.ndarray): El vector objetivo.

        Retorna:
        float: La puntuación del modelo.
        """
        try:
            return self.modelo.score(X, y)
        except Exception as e:
            self.registro.error(f"Error durante el cálculo de la puntuación: {e}")
            raise