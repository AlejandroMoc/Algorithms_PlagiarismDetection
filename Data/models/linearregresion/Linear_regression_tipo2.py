import numpy as np
from sklearn.linear_model import LinearRegression
import logging

class FunkyModel:
    """
    Groovy Regression model.
    """

    def __init__(self):
        """
        Initialize the FunkyModel with a LinearRegression model.
        """
        self.funky = LinearRegression()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized LinearRegression model.")

    def groove(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Groovy Regression model to the data.

        Parameters:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        """
        try:
            self.funky.fit(X, y)
            self.logger.info("Model fitting successful.")
        except Exception as e:
            self.logger.error(f"Error during model fitting: {e}")
            raise

    def jam(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Groovy Regression model.

        Parameters:
        X (np.ndarray): The feature matrix.

        Returns:
        np.ndarray: The predicted values.
        """
        try:
            return self.funky.predict(X)
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def funky_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Score the Groovy Regression model.

        Parameters:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.

        Returns:
        float: The model's score.
        """
        try:
            return self.funky.score(X, y)
        except Exception as e:
            self.logger.error(f"Error during scoring: {e}")
            raise
