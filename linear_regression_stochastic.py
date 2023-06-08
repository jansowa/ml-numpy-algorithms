import numpy as np
from numpy.typing import ArrayLike
import random

class LinearRegressionStochasticGradientDescent:
    _beta: ArrayLike
    @staticmethod
    def calculate_yhat_vector(X: ArrayLike, beta: ArrayLike) -> ArrayLike:
        return np.matmul(X, beta)

    @staticmethod
    def corrected_parameter(lr: float, beta: float, y: ArrayLike, y_hat: ArrayLike, x: float) -> float:
        return beta - lr * ((y_hat - y) * x)

    @staticmethod
    def squared_error(y, y_hat):
        return np.sum(np.square(y-y_hat))

    def fit(self, X: ArrayLike, y: ArrayLike, max_epochs=10000, lr=0.01):
        X_np = np.insert(np.array(X), 0, 1, axis=1)
        self._beta = np.zeros(X_np.shape[1])
        temp_beta = np.zeros(X_np.shape[1])
        y_np = np.array(y)
        for epoch in range(max_epochs):
            shuffled_X_row_idx = list(range(X_np.shape[0]))
            random.shuffle(shuffled_X_row_idx)
            for X_row_idx in shuffled_X_row_idx:
                y_hat = LinearRegressionStochasticGradientDescent.calculate_yhat_vector(X_np[X_row_idx], self._beta)
                for beta_idx in range(self._beta.shape[0]):
                    temp_beta[beta_idx] =\
                        LinearRegressionStochasticGradientDescent.corrected_parameter(lr, self._beta[beta_idx], y_np[X_row_idx], y_hat, X_np[X_row_idx, beta_idx])
                    self._beta = temp_beta

    def predict(self, X):
        X_np = np.insert(np.array(X), 0, 1, axis=1)
        return LinearRegressionStochasticGradientDescent.calculate_yhat_vector(X_np, self._beta)

X = np.array([[2, 3, 4],
              [4, 5, 6],
              [6, 7, 8],
              [7, 8, 9],
              [0, 0, 0]])
y = np.array([10, 16, 22, 25, 1])

regressor = LinearRegressionStochasticGradientDescent()
regressor.fit(X, y, max_epochs=1000, lr=0.01)
print(regressor._beta)

X_test = np.array([[4, 2, 1],
                   [0, 0, 1]])
print(regressor.predict(X_test))
print(regressor.predict(X))