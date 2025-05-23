import numpy as np
from collections import Counter

class KNearestNeighbors:
    """
    K-Nearest Neighbors classifier.

    Parameters:
    -----------
    k : int
        Number of neighbors to use for prediction.
    """

    def __init__(self, k_neighbors: int = 3):
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be greater than 0")
        self.k_neighbors = k_neighbors
        self.training_data = None
        self.training_labels = None

    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        Store the training data.
        
        Parameters:
        -----------
        data : np.ndarray
            Training data, shape (n_samples, n_features).
        labels : np.ndarray
            Training labels, shape (n_samples,).
        """
        self.training_data = data
        self.training_labels = labels

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.
        
        Parameters:
        -----------
        data : np.ndarray
            Test data, shape (n_samples, n_features).
        
        Returns:
        --------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        predictions = [self._predict_single(x) for x in data]
        return np.array(predictions)

    def _predict_single(self, sample: np.ndarray) -> int:
        """
        Predict the class label for a single test sample.
        
        Parameters:
        -----------
        sample : np.ndarray
            Single test sample, shape (n_features,).
        
        Returns:
        --------
        int
            Predicted class label.
        """
        # Calculate distances between sample and all samples in the training set
        distances = np.linalg.norm(self.training_data - sample, axis=1)
        
        # Find the k nearest neighbors and their labels
        k_indices = distances.argsort()[:self.k_neighbors]
        k_nearest_labels = self.training_labels[k_indices]
        
        # Return the most common label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common


# Example usage
if __name__ == "__main__":
    # Toy dataset
    training_data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    training_labels = np.array([0, 0, 1, 1])

    # Initialize and fit the model
    knn = KNearestNeighbors(k_neighbors=3)
    knn.fit(training_data, training_labels)

    # Predictions
    test_data = np.array([[1.5, 2.5], [3.5, 4.5]])
    predictions = knn.predict(test_data)
    print("Predictions:", predictions)