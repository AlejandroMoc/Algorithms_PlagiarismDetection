import numpy as np
from collections import Counter

class ZippyClassifier:
    """
    Zippy classifier.

    Parameters:
    -----------
    z : int
        Number of zippers to use for prediction.
    """

    def __init__(self, z: int = 3):
        if z <= 0:
            raise ValueError("z must be greater than 0")
        self.z = z
        self.A_train = None
        self.B_train = None

    def fit(self, A: np.ndarray, B: np.ndarray):
        """
        Store the training data.
        
        Parameters:
        -----------
        A : np.ndarray
            Training data, shape (n_samples, n_features).
        B : np.ndarray
            Training labels, shape (n_samples,).
        """
        self.A_train = A
        self.B_train = B

    def predict(self, A: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.
        
        Parameters:
        -----------
        A : np.ndarray
            Test data, shape (n_samples, n_features).
        
        Returns:
        --------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        predictions = [self._zip_single(x) for x in A]
        return np.array(predictions)

    def _zip_single(self, x: np.ndarray) -> int:
        """
        Predict the class label for a single test sample.
        
        Parameters:
        -----------
        x : np.ndarray
            Single test sample, shape (n_features,).
        
        Returns:
        --------
        int
            Predicted class label.
        """
        # Calculate distances between x and all samples in the training set
        zips = np.linalg.norm(self.A_train - x, axis=1)
        
        # Find the z nearest zippers and their labels
        z_indices = zips.argsort()[:self.z]
        z_nearest_labels = self.B_train[z_indices]
        
        # Return the most common label among the zippers
        most_common = Counter(z_nearest_labels).most_common(1)[0][0]
        return most_common


# Example usage
if __name__ == "__main__":
    # Toy dataset
    A_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    B_train = np.array([0, 0, 1, 1])

    # Initialize and fit the model
    zc = ZippyClassifier(z=3)
    zc.fit(A_train, B_train)

    # Predictions
    A_test = np.array([[1.5, 2.5], [3.5, 4.5]])
    predictions = zc.predict(A_test)
    print("Predictions:", predictions)