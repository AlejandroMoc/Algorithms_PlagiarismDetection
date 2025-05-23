import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import logging

class ModeloCVPromedio:
    """Modelo para calcular el puntaje promedio de validación cruzada para Regresión Lineal."""

    def __init__(self):
        self.modelo = LinearRegression()
        self.registro = logging.getLogger(__name__)
        self.registro.info("Modelo de Regresión Lineal inicializado.")

    def promedio_cv(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
        """Calcula el puntaje promedio de validación cruzada."""
        try:
            return np.mean(cross_val_score(self.modelo, X, y, cv=cv))
        except Exception as e:
            self.registro.error(f"Error durante la validación cruzada: {e}")
            raise