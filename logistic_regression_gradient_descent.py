import numpy as np
from math import exp


class LogisticRegressionGradientDescent:
    _beta: np.ndarray

    @staticmethod
    def calculate_yhat(beta: np.ndarray, X):
        return 1/(1 + exp(- np.matmul(beta.T, X)))

    @staticmethod
    def calculate_yhat_vector(beta: np.ndarray, X):
        return 1 / (1 + np.exp(np.matmul(beta, X.T)))

    @staticmethod
    def corrected_parameter(beta_j, alfa, y, y_hat, x_j):
        return beta_j + alfa * ((y - y_hat) * x_j).sum()



beta = np.array([1, 1, 1, 1])

X = np.array([[11, 10, 10, 10],
              [0, 0, 0, 0],
              [-10, -10, -10, -10]])

print(LogisticRegressionGradientDescent.corrected_parameter(10, 0.01, np.array([1, 1, 1]), np.array([1, 0, 1]), 0.5))
