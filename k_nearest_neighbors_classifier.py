from numbers import Number
import numpy as np
from numpy.typing import ArrayLike


class KNearestNeighborsClassifier:
    __X: ArrayLike
    __y: ArrayLike
    _k: int
    _classes: ArrayLike

    def fit(self, X: ArrayLike, y: ArrayLike, k) -> None:
        y_temp = np.array(y)
        if len(y_temp.shape) != 1:
            raise RuntimeError("'y' should be 1-dimension array")

        classes = np.unique(y_temp)
        if len(classes) != 2:
            raise RuntimeError("'y' should have exactly two classes")

        self.__X = X
        self.__y = y
        self._classes = classes
        self._k = k

    def predict(self, X: ArrayLike) -> ArrayLike:
        return KNearestNeighborsClassifier._predict_samples(self.__X, self.__y, X, self._k, self._classes)

    @staticmethod
    def _euclidean_distance(p1: ArrayLike | Number, p2: ArrayLike | Number) -> Number:
        return np.sqrt(np.square(np.atleast_2d(p1) - np.atleast_2d(p2)).sum(axis=1))

    @staticmethod
    def _nearest_points_args(X: ArrayLike, samples: ArrayLike | Number, k: int) -> ArrayLike:
        return np.array(
            [np.argsort(KNearestNeighborsClassifier._calculate_distances(X, sample))[:k] for sample in samples])

    @staticmethod
    def _calculate_distances(points_arr: ArrayLike, point: ArrayLike | Number) -> ArrayLike:
        points = np.array(points_arr)
        return KNearestNeighborsClassifier._euclidean_distance(points, point)

    @staticmethod
    def _transform_to_class(mean_value: float, classes: ArrayLike) -> Number:
        return classes[np.argsort(np.abs(classes - mean_value))[0]]

    @staticmethod
    def _predict_samples(X: ArrayLike, y: ArrayLike, samples: ArrayLike | Number, k: int,
                         classes: ArrayLike) -> ArrayLike | Number:
        nearest_args = KNearestNeighborsClassifier._nearest_points_args(X, samples, k)
        means = np.array(y)[nearest_args].sum(axis=1) / k
        return np.array([KNearestNeighborsClassifier._transform_to_class(prediction, classes) for prediction in means])
