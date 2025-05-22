import numpy as np
from catboost import CatBoostRegressor
import logging

class CatBoostModel:
    """
    CatBoost regression model.
    """
    def __init__(self):
        self.model_cb = CatBoostRegressor(silent=True)
        self.logger_cb = logging.getLogger(__name__)
        self.logger_cb.info("Initialized CatBoostRegressor.")

    def fit_model(self, X_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Fit the CatBoost model to the data.
        """
        try:
            self.model_cb.fit(X_data, y_data)
            self.logger_cb.info("Model fitting successful.")
        except Exception as exc:
            self.logger_cb.error(f"Error during model fitting: {exc}")
            raise

    def predict_values(self, X_data: np.ndarray) -> np.ndarray:
        """
        Predict using the CatBoost model.
        """
        try:
            return self.model_cb.predict(X_data)
        except Exception as exc:
            self.logger_cb.error(f"Error during prediction: {exc}")
            raise

    def calculate_score(self, X_data: np.ndarray, y_data: np.ndarray) -> float:
        """
        Score the CatBoost model.
        """
        try:
            return self.model_cb.score(X_data, y_data)
        except Exception as exc:
            self.logger_cb.error(f"Error during scoring: {exc}")
            raise