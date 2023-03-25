import numpy as np
from numpy.typing import ArrayLike

def calculate_beta(X: ArrayLike, y: ArrayLike):
    X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
    X_ones = np.insert(X_temp, 0, 1, axis=1)
    return np.matmul(
        np.matmul(
            np.linalg.inv(
                np.matmul(X_ones.T, X_ones)),
            X_ones.T),
        y)
