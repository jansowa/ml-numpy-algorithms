import numpy as np
from math import exp
from numpy.typing import ArrayLike


class LogisticRegressionGradientDescent:
    _beta: np.ndarray

    @staticmethod
    def calculate_yhat(beta: np.ndarray, X):
        return 1/(1 + exp(- beta.T.dot(X)))

    @staticmethod
    def calculate_yhat_vector(beta: np.ndarray, X):
        return 1 / (1 + np.exp(-beta.dot(X.T)))

    @staticmethod
    def corrected_parameter(beta_j, lr, y, y_hat, x_j):
        return beta_j + lr * ((y - y_hat) * x_j).sum()

    def fit(self, X: ArrayLike, y: ArrayLike, max_epochs=10000, lr=0.01):
        X_np = np.insert(np.array(X), 0, 1, axis=1)
        self._beta = np.zeros(X_np.shape[1])
        y_np = np.array(y)
        y_hat = LogisticRegressionGradientDescent.calculate_yhat_vector(self._beta, X_np)
        for _ in range(max_epochs):
            self._beta = self._beta - lr / X_np.shape[0] * X_np.T @ (y_hat - y_np)
            y_hat = LogisticRegressionGradientDescent.calculate_yhat_vector(self._beta, X_np)

    def predict(self, X):
        X_np = np.insert(np.array(X), 0, 1, axis=1)
        return LogisticRegressionGradientDescent.calculate_yhat_vector(self._beta, X_np)


X = np.array([[3, 4],
              [2, 2],
              [-1, -1],
              [-2, -2],
              [-2, -3],
              [3, 3]])
y = np.array([1, 1, 0, 0, 0, 1])

model = LogisticRegressionGradientDescent()
model.fit(X, y, max_epochs=1000)
print(model._beta)
print(model.predict(X))

X_test = np.array([[1, 2],
                   [2, 1],
                   [3, 1],
                   [1, 3],
                   [-1, -2],
                   [-2, -1],
                   [-3, -1],
                   [-1, -3]])
print(model.predict(X_test))