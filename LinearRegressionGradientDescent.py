import numpy as np
from numpy.typing import ArrayLike

class LinearRegressionGradientDescent:
    _beta: ArrayLike
    @staticmethod
    def calculate_yhat_vector(X: ArrayLike, beta: ArrayLike) -> ArrayLike:
        return np.matmul(X, beta)

    @staticmethod
    def corrected_parameter(lr: float, beta: float, y: ArrayLike, y_hat: ArrayLike, X_vec: ArrayLike) -> float:
        return beta - (lr / y.shape[0]) * ((y_hat - y) * X_vec).sum()

    @staticmethod
    def gradient_iteration(lr: float, beta_vec: ArrayLike, y: ArrayLike, X: ArrayLike):
        y_hat = LinearRegressionGradientDescent.calculate_yhat_vector(X, beta_vec)
        for i in range(beta_vec.shape[0]):
            beta_vec[i] = LinearRegressionGradientDescent.corrected_parameter(lr, beta_vec[i], y, y_hat, X[:, i])
        return beta_vec

    def fit(self, X: ArrayLike, y: ArrayLike, epochs=10000, lr=0.01):
        X_np = np.array(X)
        self._beta = np.zeros(X_np.shape[1])
        y_np = np.array(y)
        for i in range(epochs):
            self._beta = LinearRegressionGradientDescent.gradient_iteration(lr, self._beta, y_np, X_np)

    def predict(self, X):
        return LinearRegressionGradientDescent.calculate_yhat_vector(np.array(X), self._beta)

X = np.array([[1, 2, 3, 4],
              [1, 4, 5, 6],
              [1, 6, 7, 8],
              [1, 7, 8, 9],
              [1, 0, 0, 0]])
y = np.array([10, 16, 22, 25, 1])

regressor = LinearRegressionGradientDescent()
regressor.fit(X, y)
print(regressor._beta)

X_test = np.array([[0, 4, 2, 1],
                   [0, 0, 0, 1]])
print(regressor.predict(X_test))