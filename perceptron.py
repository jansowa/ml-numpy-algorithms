import numpy as np
from numpy.typing import ArrayLike

class Perceptron:
    @staticmethod
    def activation_function(X: ArrayLike, weights: ArrayLike) -> float:
        X = np.array(X)
        weights = np.array(weights)
        return np.matmul(np.append(X, 1), weights.T)

    @staticmethod
    def step_function(activation: float) -> int:
        return 1 if activation >= 0 else 0
