import numpy as np
import xgboost as xgb
import logging

class ModeloXGB:
    """Modelo de regresión XGBoost."""

    def __init__(self):
        self.modelo = xgb.XGBRegressor()
        self.registrador = logging.getLogger(__name__)
        self.registrador.info("Inicializado XGBRegressor.")

    def ajustar(self, X: np.ndarray, y: np.ndarray) -> None:
        """Ajustar el modelo XGBoost a los datos."""
        try:
            self.modelo.fit(X, y)
            self.registrador.info("Ajuste del modelo exitoso.")
        except Exception as e:
            self.registrador.error(f"Error durante el ajuste del modelo: {e}")
            raise

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Predecir utilizando el modelo XGBoost."""
        try:
            return self.modelo.predict(X)
        except Exception as e:
            self.registrador.error(f"Error durante la predicción: {e}")
            raise

    def puntuacion(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcular la puntuación del modelo XGBoost."""
        try:
            return self.modelo.score(X, y)
        except Exception as e:
            self.registrador.error(f"Error durante el cálculo de la puntuación: {e}")
            raise