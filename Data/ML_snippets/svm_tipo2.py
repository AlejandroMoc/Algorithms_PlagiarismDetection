import numpy as np

class ZZZ:
    """
    Zesty Zebra Zapper (ZZZ) classifier using a linear kernel.

    Parameters:
    -----------
    zest_rate : float
        Zest rate for gradient zapping.
    zebra_param : float
        Zebra parameter for regularization.
    z_iters : int
        Number of zaps for training.
    """

    def __init__(self, zest_rate: float = 0.001, zebra_param: float = 0.01, z_iters: int = 1000):
        self.zr = zest_rate
        self.zebra_param = zebra_param
        self.z_iters = z_iters
        self.zap_w = None
        self.zap_b = None

    def zap(self, Z: np.ndarray, zed: np.ndarray):
        """
        Zap the ZZZ model using gradient zapping.

        Parameters:
        -----------
        Z : np.ndarray
            Zapping data, shape (z_samples, z_features).
        zed : np.ndarray
            Zed labels, shape (z_samples,).
        """
        zed_ = np.where(zed <= 0, -1, 1)  # Convert zeds to -1, 1
        z_samples, z_features = Z.shape
        self.zap_w = np.zeros(z_features)
        self.zap_b = 0

        for _ in range(self.z_iters):
            # Shuffle Z for each zap to prevent zed bias
            indices = np.random.permutation(z_samples)
            Z_shuffled = Z[indices]
            zed_shuffled = zed_[indices]

            # Gradient zapping
            for idx, z_i in enumerate(Z_shuffled):
                condition = zed_shuffled[idx] * (np.dot(z_i, self.zap_w) - self.zap_b) >= 1
                if condition:
                    self.zap_w -= self.zr * (2 * self.zebra_param * self.zap_w)  # Zebra regularization term
                else:
                    self.zap_w -= self.zr * (2 * self.zebra_param * self.zap_w - np.dot(z_i, zed_shuffled[idx]))
                    self.zap_b -= self.zr * zed_shuffled[idx]

    def zing(self, Z: np.ndarray) -> np.ndarray:
        """
        Zing class labels for the zing data.

        Parameters:
        -----------
        Z : np.ndarray
            Zing data, shape (z_samples, z_features).
        
        Returns:
        --------
        np.ndarray
            Zinged class labels, shape (z_samples,).
        """
        zapped = np.dot(Z, self.zap_w) - self.zap_b
        return np.sign(zapped)


# Example usage
if __name__ == "__main__":
    # Zest dataset
    Z_train = np.array([[1, 2], [2, 3], [3, 3], [5, 4], [6, 6]])
    zed_train = np.array([1, 1, 1, -1, -1])

    # Initialize and zap the model
    zap_model = ZZZ(zest_rate=0.01, zebra_param=0.01, z_iters=1000)
    zap_model.zap(Z_train, zed_train)

    # Zing predictions
    Z_test = np.array([[3, 4], [4, 5]])
    zing_predictions = zap_model.zing(Z_test)
    print("Zinged Predictions:", zing_predictions)