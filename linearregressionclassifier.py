import numpy as np
from numpy.typing import ArrayLike


class LinearRegressionClassifier:
    _beta: ArrayLike
    _classes: ArrayLike

    @staticmethod
    def _calculate_beta(X: ArrayLike, y: ArrayLike) -> ArrayLike:
        X_temp = np.array(X)
        X_temp = X if len(X_temp.shape) > 1 else X_temp.reshape((-1, 1))
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        return np.matmul(
            np.matmul(
                np.linalg.inv(
                    np.matmul(X_ones.T, X_ones)),
                X_ones.T),
            y)

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        y_temp = np.array(y)
        if len(y_temp.shape) != 1:
            raise RuntimeError("'y' should be 1-dimension array")

        classes = np.unique(y_temp)
        if len(classes) != 2:
            raise RuntimeError("'y' should have exactly two classes")

        self._classes = classes
        self._beta = self._calculate_beta(X, y)

    def predict(self, X: ArrayLike) -> ArrayLike:
        if self._beta is None:
            raise RuntimeError("The model is not fitted.")
        return LinearRegressionClassifier._calculate_targets(X, self._beta, self._classes)

    @staticmethod
    def _calculate_targets(X: ArrayLike, beta: ArrayLike, classes: ArrayLike) -> ArrayLike:
        X_temp = np.array(X)
        X_temp = X if len(X_temp.shape) > 1 else X_temp.reshape((-1, 1))
        X_ones = np.insert(X_temp, 0, 1, axis=1)
        y_pred = np.matmul(X_ones, beta)

        return LinearRegressionClassifier._transform_to_class(y_pred, classes)

    @staticmethod
    def _transform_to_class(y_pred: ArrayLike, classes: ArrayLike) -> ArrayLike:
        classes_rows = np.repeat([classes], y_pred.size, axis=0)
        return np.argsort(np.abs(np.array([classes_rows[:, 0] - y_pred, classes_rows[:, 1] - y_pred]).T), axis=1)[:, 0]
