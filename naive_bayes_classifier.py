import numpy as np


class NaiveBayesClassifier:

    @staticmethod
    def calculate_theta_y(y: np.ndarray) -> float:
        return y.sum() / len(y)

    @staticmethod
    def calculate_theta_j_y1(X: np.ndarray, y: np.ndarray, j: int) -> float:
        X_j = X[:, j]
        return len(X_j[(y == 1) & (X_j == 1)]) / y.sum()

    @staticmethod
    def calculate_theta_j_y0(X: np.ndarray, y: np.ndarray, j: int) -> float:
        X_j = X[:, j]
        return len(X_j[(y == 1) & (X_j == 1)]) / (len(y) - y.sum())


y = np.array([1, 1, 0, 1, 1, 1])

X = np.array([[1, 1, 1, 1],
              [0, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [1, 1, 0, 0],
              [1, 0, 1, 1]])

print(len(y))

# print(X[:, 2])

X_j = X[:, 2]
print(X_j)
print(y)
print(X_j[(y == 1) & (X_j == 1)])
print(NaiveBayesClassifier.calculate_theta_j_y1(X, y, 2))
print(NaiveBayesClassifier.calculate_theta_j_y0(X, y, 2))