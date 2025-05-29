import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import logging

class CVMM:
    """Model to compute cross-validation mean score for Linear Regression."""

    def __init__(self):
        self.mmm = LinearRegression()
        self.loggy = logging.getLogger(__name__)
        self.loggy.info("Initialized LinearRegression model.")

    def cv_mean(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
        """Compute the mean cross-validation score."""
        try:
            return np.mean(cross_val_score(self.mmm, X, y, cv=cv))
        except Exception as e:
            self.loggy.error(f"Error during cross-validation: {e}")
            raise