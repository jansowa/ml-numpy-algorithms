import numpy as np
from numpy.typing import ArrayLike


class WeightedLinearRegressionNormalEquation:
    _beta = None
    _weights = None

    @staticmethod
    def _calculate_beta(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> ArrayLike:
        W = np.diag(weights)
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        return np.matmul(
            np.matmul(
                np.matmul(
                    np.linalg.inv(
                        np.matmul(
                            np.matmul(
                                X_ones.T,
                                W),
                            X_ones)),
                    X_ones.T),
                W),
            y)

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        self._beta = self._calculate_beta(X, y, weights)

    def predict(self, X: np.ndarray) -> ArrayLike:
        if self._beta is None:
            raise RuntimeError("The model is not fitted.")
        return WeightedLinearRegressionNormalEquation._calculate_targets(X, self._beta)

    @staticmethod
    def _calculate_targets(X: np.ndarray, beta: np.ndarray) -> ArrayLike:
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        return np.matmul(X_ones, beta)
