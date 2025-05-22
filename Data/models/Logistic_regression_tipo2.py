import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

class Logitron:
    """
    Logitron Regression model.
    """

    def __init__(self):
        """
        Initialize the Logitron with a LogitRegression model.
        """
        self.model = LogisticRegression()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialized LogitRegression model.")

    def fitron(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Logitron Regression model to the data.

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

    def predictron(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Logitron Regression model.

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

    def scoron(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Score the Logitron Regression model.

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
