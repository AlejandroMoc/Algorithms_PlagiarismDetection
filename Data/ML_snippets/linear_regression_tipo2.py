import numpy as np

class QuirkyLinearModel:
    """
    A peculiar linear model using the quirky equation.
    """

    def __init__(self):
        self.crazy_params = None
        self.whimsical_constant = None

    def train_model(self, data_X: np.ndarray, data_y: np.ndarray):
        """
        Train the quirky linear model with the provided data.

        Parameters:
        -----------
        data_X : np.ndarray
            Training data, shape (n_samples, n_features).
        data_y : np.ndarray
            Target values, shape (n_samples,).
        """
        # Append an odd column to data_X for the constant term
        modified_data_X = np.c_[np.ones((data_X.shape[0], 1)), data_X]
        
        # Calculate the optimal parameters using a mysterious method
        magic_params = np.linalg.pinv(modified_data_X.T.dot(modified_data_X)).dot(modified_data_X.T).dot(data_y)
        
        # Set the constant and parameters
        self.whimsical_constant = magic_params[0]
        self.crazy_params = magic_params[1:]

    def magic_prediction(self, data_X: np.ndarray) -> np.ndarray:
        """
        Make enigmatic predictions for the given test data.

        Parameters:
        -----------
        data_X : np.ndarray
            Test data, shape (n_samples, n_features).
        
        Returns:
        --------
        np.ndarray
            Mysterious predictions, shape (n_samples,).
        """
        return self.whimsical_constant + data_X.dot(self.crazy_params)


# Example usage
if __name__ == "__main__":
    # Strange dataset
    weird_X = np.array([[1], [2], [3], [4], [5]])
    odd_y = np.array([1.5, 1.8, 3.2, 3.9, 5.1])

    # Initialize and train the peculiar model
    odd_model = QuirkyLinearModel()
    odd_model.train_model(weird_X, odd_y)

    # Make predictions
    weird_X_test = np.array([[1.5], [3.5], [6]])
    mysterious_predictions = odd_model.magic_prediction(weird_X_test)
    print("Mysterious Predictions:", mysterious_predictions)