import numpy as np
from numpy.typing import ArrayLike


class WeightedLinearRegressionNormalEquation:
    _beta: np.ndarray
    _weights: np.ndarray

    @staticmethod
    def _calculate_beta(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        W = np.diag(weights)
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        return np.linalg.inv(X_ones.T.dot(W).dot(X_ones)) \
            .dot(X_ones.T).dot(W).dot(y)

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
        return X_ones.dot(beta)
