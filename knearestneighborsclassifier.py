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
        a1 = np.array(p1)
        a2 = np.array(p2)
        return np.sqrt(np.square(a1 - a2).sum())

    @staticmethod
    def _nearest_points(points_arr: ArrayLike, point: ArrayLike | Number, k: int) -> ArrayLike:
        distances = KNearestNeighborsClassifier._calculate_distances(points_arr, point)
        return np.array(points_arr)[np.argsort(distances)[:k]]

    @staticmethod
    def _nearest_points_args(points_arr: ArrayLike, point: ArrayLike | Number, k: int) -> ArrayLike:
        distances = KNearestNeighborsClassifier._calculate_distances(points_arr, point)
        return np.argsort(distances)[:k]

    # TODO: vectorize method
    @staticmethod
    def _calculate_distances(points_arr: ArrayLike, point: ArrayLike | Number) -> ArrayLike:
        points = np.array(points_arr)
        result = []
        for idx in range(points.shape[0]):
            distance = KNearestNeighborsClassifier._euclidean_distance(points[idx], point)
            result += [distance]

        return np.array(result)

    @staticmethod
    def _predict_sample(X: ArrayLike, y: ArrayLike, sample: ArrayLike | Number, k: int, classes: ArrayLike) -> Number:
        nearest_args = KNearestNeighborsClassifier._nearest_points_args(X, sample, k)
        mean_value = np.array(y)[nearest_args].sum() / k
        return KNearestNeighborsClassifier._transform_to_class(mean_value, classes)

    @staticmethod
    def _transform_to_class(mean_value: float, classes: ArrayLike) -> Number:
        return classes[np.argsort(np.abs(classes - mean_value))[0]]

    # TODO: vectorize
    @staticmethod
    def _predict_samples(X: ArrayLike, y: ArrayLike, samples: ArrayLike | Number, k: int, classes: ArrayLike) -> ArrayLike | Number:
        result = []
        for sample in samples:
            result += [(KNearestNeighborsClassifier._predict_sample(X, y, sample, k, classes))]
        return np.array(result)
