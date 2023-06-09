import numpy as np
from math import exp
from numpy.typing import ArrayLike


class PoissonRegression:
    _beta: np.ndarray

    @staticmethod
    def calculate_yhat(beta: np.ndarray, X):
        return exp(beta.T.dot(X))

    @staticmethod
    def calculate_yhat_vector(beta: np.ndarray, X):
        return np.exp(beta.dot(X.T))

    @staticmethod
    def corrected_parameter(beta_j, lr, y, y_hat, x_j):
        return beta_j + lr * ((y - y_hat) * x_j).sum()

    def fit(self, X: ArrayLike, y: ArrayLike, max_epochs=10000, lr=0.01):
        X_np = np.insert(np.array(X), 0, 1, axis=1)
        self._beta = np.zeros(X_np.shape[1])
        temp_beta = np.zeros(X_np.shape[1])
        y_np = np.array(y)
        y_hat = PoissonRegression.calculate_yhat_vector(self._beta, X_np)
        for _ in range(max_epochs):
            for beta_idx in range(self._beta.shape[0]):
                temp_beta[beta_idx] =\
                    PoissonRegression.corrected_parameter(
                        temp_beta[beta_idx], lr, y_np, y_hat, X_np[:, beta_idx]
                    )

            self._beta = temp_beta
            y_hat = PoissonRegression.calculate_yhat_vector(self._beta, X_np)
    def predict(self, X):
        X_np = np.insert(np.array(X), 0, 1, axis=1)
        return PoissonRegression.calculate_yhat_vector(self._beta, X_np)


# price = 10 + 2x1 + 3x2 - 1x3

y = [4, 8, 15, 12, 35]
X = [[0.1, 0.2, 0.3],
     [0.3, 0.2, 0.1],
     [0.13, 0.62, 0.42],
     [0.37, 0.33, 0.27],
     [0.84, 0.37, 0.24]]

pr = PoissonRegression()
pr.fit(X, y, max_epochs=100000)
print(pr._beta)
print(pr.predict(X))
# def help(x1, x2, x3):
#     return exp(1 + 2*x1 + 3*x2 - 1*x3)
#
# print(help(0.84, 0.37, 0.24))