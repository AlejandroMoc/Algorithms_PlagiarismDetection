import numpy as np
from sklearn.cluster import KMeans
import logging

class ClustModel:
    """
    Clustering model.
    """

    def __init__(self, num_clusters: int = 3):
        """
        Initialize the ClustModel with a specified number of clusters.

        Parameters:
        num_clusters (int): The number of clusters to form and the number of centroids to generate.
        """
        self.mod = KMeans(n_clusters=num_clusters)
        self.log = logging.getLogger(__name__)
        self.log.info(f"Initialized ClustModel with num_clusters={num_clusters}.")

    def fit_data(self, Y: np.ndarray) -> None:
        """
        Fit the ClustModel to the data.

        Parameters:
        Y (np.ndarray): The feature matrix.
        """
        try:
            self.mod.fit(Y)
            self.log.info("Model fitting successful.")
        except Exception as exc:
            self.log.error(f"Error during model fitting: {exc}")
            raise

    def make_prediction(self, Y: np.ndarray) -> np.ndarray:
        """
        Predict cluster indices for samples.

        Parameters:
        Y (np.ndarray): The feature matrix.

        Returns:
        np.ndarray: The predicted cluster indices.
        """
        try:
            return self.mod.predict(Y)
        except Exception as exc:
            self.log.error(f"Error during prediction: {exc}")
            raise