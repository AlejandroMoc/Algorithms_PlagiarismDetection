import numpy as np
import lightgbm as lgb
import logging

class SparklyModel:
    """Sparkly regression model."""
    def __init__(self):
        self.sparkle = lgb.LGBMRegressor()
        self.glitter = logging.getLogger(__name__)
        self.glitter.info("Initialized LGBMRegressor.")

    def dazzle(self, X: np.ndarray, y: np.ndarray) -> None:
        """Dazzle the Sparkly model with the data."""
        try:
            self.sparkle.fit(X, y)
            self.glitter.info("Model fitting successful.")
        except Exception as e:
            self.glitter.error(f"Error during model fitting: {e}")
            raise

    def twinkle(self, X: np.ndarray) -> np.ndarray:
        """Twinkle using the Sparkly model."""
        try:
            return self.sparkle.predict(X)
        except Exception as e:
            self.glitter.error(f"Error during prediction: {e}")
            raise

    def shine(self, X: np.ndarray, y: np.ndarray) -> float:
        """Shine the Sparkly model."""
        try:
            return self.sparkle.score(X, y)
        except Exception as e:
            self.glitter.error(f"Error during scoring: {e}")
            raise