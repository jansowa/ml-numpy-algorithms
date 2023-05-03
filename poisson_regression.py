import numpy as np
from math import exp


class PoissonRegression:
    _beta: np.ndarray

    @staticmethod
    def calculate_yhat(beta: np.ndarray, X):
        return exp(np.matmul(beta.T, X))

    @staticmethod
    def calculate_yhat_vector(beta: np.ndarray, X):
        return np.exp(np.matmul(beta, X.T))

    def predict(self, X):
        X_np = np.insert(np.array(X), 0, 1, axis=1)
        return PoissonRegression.calculate_yhat_vector(self._beta, X_np)
