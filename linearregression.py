import numpy as np
from numpy.typing import ArrayLike

class LinearRegression:
    beta = None
    @staticmethod
    def calculate_beta(X: ArrayLike, y: ArrayLike):
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        return np.matmul(
            np.matmul(
                np.linalg.inv(
                    np.matmul(X_ones.T, X_ones)),
                X_ones.T),
            y)

    def fit(self, X: ArrayLike, y: ArrayLike):
        self.beta = self.calculate_beta(X, y)

    def predict(self, X: ArrayLike):
        if self.beta is None:
            raise RuntimeError("The model is not fitted.")
        return LinearRegression.calculate_targets(X, self.beta)

    @staticmethod
    def calculate_targets(X: ArrayLike, beta: ArrayLike):
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        return np.matmul(X_ones, beta)
