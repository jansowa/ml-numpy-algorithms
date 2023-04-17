import numpy as np
from numpy.typing import ArrayLike


class Perceptron:
    _weights: ArrayLike = None

    def fit(self, X, y, learning_rate: float = 0.1, epochs: int = 500):
        self._weights = np.ones(X.shape[1] + 1)
        for _ in range(epochs):
            self._weights = Perceptron.single_epoch(X, self._weights, y, learning_rate=learning_rate)

    def predict(self, X):
        return Perceptron.make_prediction(X, self._weights)

    @staticmethod
    def activation_function(X: ArrayLike, weights: ArrayLike) -> ArrayLike:
        X_ones = np.insert(X, 0, 1, axis=1)
        weights = np.array(weights)
        return np.matmul(X_ones, weights.T)

    @staticmethod
    def step_function(activation: ArrayLike) -> ArrayLike:
        def single_step(single_activation: float):
            return 1 if single_activation >= 0 else 0

        return np.vectorize(single_step)(activation)

    @staticmethod
    def make_prediction(X: ArrayLike, weights: ArrayLike) -> ArrayLike:
        return Perceptron.step_function(Perceptron.activation_function(X, weights))

    @staticmethod
    def single_epoch(X: ArrayLike, weights: ArrayLike, y: ArrayLike, learning_rate: float = 0.01):
        y_pred = Perceptron.make_prediction(X, weights)
        return weights + np.matmul(learning_rate * (y - y_pred), np.insert(X, 0, 1, axis=1))


X = np.array([[1, 0, 1],
              [1, 1, 0],
              [0, 1, 1],
              [0, 0, 0],
              [1, 1, 1],
              [0, 0, 1]])
y = [0, 1, 1, 1, 1, 0]
p = Perceptron()
p.fit(X, y)
print(p._weights)
print(p.predict(X))
