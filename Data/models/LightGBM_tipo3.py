import numpy as np
import lightgbm as lgb
import logging

class LightGBMModel:
    """LightGBM regression model."""
    def __init__(self):
        self.modelo = lgb.LGBMRegressor()
        self.registro = logging.getLogger(__name__)
        self.registro.info("Inicializado LGBMRegressor.")

    def ajustar(self, X: np.ndarray, y: np.ndarray) -> None:
        """Ajusta el modelo LightGBM a los datos."""
        try:
            self.modelo.fit(X, y)
            self.registro.info("Ajuste del modelo exitoso.")
        except Exception as e:
            self.registro.error(f"Error durante el ajuste del modelo: {e}")
            raise

    def predecir(self, X: np.ndarray) -> np.ndarray:
        """Predice usando el modelo LightGBM."""
        try:
            return self.modelo.predict(X)
        except Exception as e:
            self.registro.error(f"Error durante la predicción: {e}")
            raise

    def puntuacion(self, X: np.ndarray, y: np.ndarray) -> float:
        """Puntuación del modelo LightGBM."""
        try:
            return self.modelo.score(X, y)
        except Exception as e:
            self.registro.error(f"Error durante la puntuación: {e}")
            raise