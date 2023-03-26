from numbers import Number
import numpy as np
from numpy.typing import ArrayLike


class KNearestNeighbors:
    @staticmethod
    def _euclidean_distance(p1: ArrayLike | Number, p2: ArrayLike | Number) -> Number:
        a1 = np.array(p1)
        a2 = np.array(p2)
        return np.sqrt(np.square(a1 - a2).sum())

    @staticmethod
    def _nearest_points(points_arr: ArrayLike, point: ArrayLike | Number, k: int) -> ArrayLike:
        distances = KNearestNeighbors._calculate_distances(points_arr, point)
        return np.array(points_arr)[np.argsort(distances)[:k]]

    @staticmethod
    def _nearest_points_args(points_arr: ArrayLike, point: ArrayLike | Number, k: int):
        distances = KNearestNeighbors._calculate_distances(points_arr, point)
        return np.argsort(distances)[:k]

    # TODO: vectorize method
    @staticmethod
    def _calculate_distances(points_arr: ArrayLike, point: ArrayLike | Number) -> ArrayLike:
        points = np.array(points_arr)
        result = []
        for idx in range(points.shape[0]):
            distance = KNearestNeighbors._euclidean_distance(points[idx], point)
            result += [distance]

        return np.array(result)

    @staticmethod
    def _predict_sample(X: ArrayLike, y: ArrayLike, sample: ArrayLike | Number, k: int) -> Number:
        nearest_args = KNearestNeighbors._nearest_points_args(X, sample, k)
        return np.array(y)[nearest_args].sum() / k

    # TODO: vectorize
    @staticmethod
    def _predict_samples(X: ArrayLike, y: ArrayLike, samples: ArrayLike | Number, k: int) -> ArrayLike | Number:
        result = []
        for sample in samples:
            result += [(KNearestNeighbors._predict_sample(X, y, sample, k))]
        return np.array(result)
