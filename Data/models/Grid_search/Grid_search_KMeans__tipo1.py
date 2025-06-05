import numpy as np
from sklearn.cluster import KMeans
import logging

class ModelKMeans:
    """
    KMeans clustering model.
    """

    def __init__(self, n_clusters: int = 3):
        """
        Initialize the ModelKMeans with a specified number of clusters.

        Parameters:
        n_clusters (int): The number of clusters to form and the number of centroids to generate.
        """
        self.model = KMeans(n_clusters=n_clusters)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ModelKMeans with n_clusters={n_clusters}.")

    def fit_model(self, data: np.ndarray) -> None:
        """
        Fit the ModelKMeans model to the data.

        Parameters:
        data (np.ndarray): The feature matrix.
        """
        try:
            self.model.fit(data)
            self.logger.info("Model fitting successful.")
        except Exception as error:
            self.logger.error(f"Error during model fitting: {error}")
            raise

    def predict_clusters(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster indices for samples.

        Parameters:
        data (np.ndarray): The feature matrix.

        Returns:
        np.ndarray: The predicted cluster indices.
        """
        try:
            return self.model.predict(data)
        except Exception as error:
            self.logger.error(f"Error during prediction: {error}")
            raise