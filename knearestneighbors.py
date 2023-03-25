from numbers import Number
import numpy as np
from numpy.typing import ArrayLike

class KNearestNeighbors:
    @staticmethod
    def euclidean_distance(p1: ArrayLike | Number, p2: ArrayLike | Number) -> Number:
        a1 = np.array(p1)
        a2 = np.array(p2)
        return np.sqrt(np.square(a1 - a2).sum())