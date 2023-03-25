from numbers import Number
import numpy as np
from numpy.typing import ArrayLike

class KNearestNeighbors:
    @staticmethod
    def euclidean_distance(p1: ArrayLike | Number, p2: ArrayLike | Number) -> Number:
        a1 = np.array(p1)
        a2 = np.array(p2)
        return np.sqrt(np.square(a1 - a2).sum())

    @staticmethod
    def nearest_points(points_arr: ArrayLike, point: ArrayLike | Number, k: int) -> ArrayLike:
        distances = KNearestNeighbors.calculate_distances(points_arr, point)
        return np.array(points_arr)[np.argsort(distances)[:k]]

    # TODO: vectorize method
    @staticmethod
    def calculate_distances(points_arr: ArrayLike, point: ArrayLike | Number) -> ArrayLike:
        points = np.array(points_arr)
        result = []
        for idx in range(points.shape[0]):
            distance = KNearestNeighbors.euclidean_distance(points[idx], point)
            result += [distance]

        return np.array(result)
