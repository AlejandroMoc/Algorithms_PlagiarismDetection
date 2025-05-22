import numpy as np
from catboost import CatBoostRegressor
import logging

class WhiskerBoostModel:
    """
    WhiskerBoost regression model.
    """
    def __init__(self):
        self.model = CatBoostRegressor(silent=True)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized CatBoostRegressor.")

    def train_fur(self, P: np.ndarray, Q: np.ndarray) -> None:
        """
        Train the WhiskerBoost model with the data.
        """
        try:
            self.model.fit(P, Q)
            self.logger.info("Model training successful.")
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise

    def whisk_predict(self, P: np.ndarray) -> np.ndarray:
        """
        Predict using the WhiskerBoost model.
        """
        try:
            return self.model.predict(P)
        except Exception as e:
            self.logger.error(f"Error during whisker prediction: {e}")
            raise

    def fur_score(self, P: np.ndarray, Q: np.ndarray) -> float:
        """
        Score the WhiskerBoost model.
        """
        try:
            return self.model.score(P, Q)
        except Exception as e:
            self.logger.error(f"Error during scoring the fur: {e}")
            raise