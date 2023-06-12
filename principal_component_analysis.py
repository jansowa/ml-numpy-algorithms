import numpy as np


class PCA:
    @staticmethod
    def standardize_data(X: np.ndarray) -> np.ndarray:
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def fit(self, X: np.ndarray):
        X_normalized = PCA.standardize_data(X)
        X_cov = np.cov(X_normalized)
        eigval, eigvec = np.linalg.eig(X_cov)


X = np.array([[1, 2, 3],
              [4, 3, 2],
              [-1, -3, -4],
              [-2, 5, -2],
              [-10, 10, 2]])

X_normalized = PCA.standardize_data(X)
print(np.cov(X_normalized))
