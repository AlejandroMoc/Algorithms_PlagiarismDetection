import numpy as np
from sklearn.linear_model import LinearRegression
import logging

class ModeloLineal:
    """
    Modelo de regresión lineal.
    """

    def __init__(self):
        """
        Inicializa el ModeloLineal con un modelo de regresión lineal.
        """
        self.modelo = LinearRegression()
        self.registrador = logging.getLogger(__name__)
        self.registrador.info("Modelo de regresión lineal inicializado.")

    def ajustar(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta el modelo de regresión lineal a los datos.

        Parámetros:
        X (np.ndarray): La matriz de características.
        y (np.ndarray): El vector objetivo.
        """
        try:
            self.modelo.fit(X, y)
            self.registrador.info("Ajuste del modelo exitoso.")
        except Exception as e:
            self.registrador.error(f"Error durante el ajuste del modelo: {e}")
            raise

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """
        Predice usando el modelo de regresión lineal.

        Parámetros:
        X (np.ndarray): La matriz de características.

        Retorna:
        np.ndarray: Los valores predichos.
        """
        try:
            return self.modelo.predict(X)
        except Exception as e:
            self.registrador.error(f"Error durante la predicción: {e}")
            raise

    def puntuacion(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula la puntuación del modelo de regresión lineal.

        Parámetros:
        X (np.ndarray): La matriz de características.
        y (np.ndarray): El vector objetivo.

        Retorna:
        float: La puntuación del modelo.
        """
        try:
            return self.modelo.score(X, y)
        except Exception as e:
            self.registrador.error(f"Error durante el cálculo de la puntuación: {e}")
            raise