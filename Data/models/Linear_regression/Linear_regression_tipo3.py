import numpy as np
from sklearn.linear_model import LinearRegression
import logging

class LinearModel:
    """
    Linear Regression model.
    """

    
    
        Parameters:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        """
        try:
            self.model.fit(X, y)
            self.logger.info("Model fitting successful.")
        except Exception as e:
            self.logger.error(f"Error during model fitting: {e}")
            raise

    
        Parameters:
        X (np.ndarray): The feature matrix.

        Returns:
        np.ndarray: The predicted values.
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    
        Parameters:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.

        Returns:
        float: The model's score.
        """
        try:
            return self.model.score(X, y)
        except Exception as e:
            self.logger.error(f"Error during scoring: {e}")
            raise


def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Score the Linear Regression model.


def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Linear Regression model.


def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Linear Regression model to the data.


def __init__(self):
        """
        Initialize the LinearModel with a LinearRegression model.
        """
        self.model = LinearRegression()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized LinearRegression model.")
