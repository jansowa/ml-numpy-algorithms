import math
import unittest
import numpy as np
from knearestneighbors import *

class TestKNearestNeighbors(unittest.TestCase):

    def test_calculate_distance_r1(self):
        """Calculate distance for points in R^1"""
        p1 = 2
        p2 = 4
        self.assertAlmostEqual(KNearestNeighbors.euclidean_distance(p1, p2), 2)

        p3 = -2
        p4 = 2
        self.assertAlmostEqual(KNearestNeighbors.euclidean_distance(p3, p4), 4)

        p5 = -2
        p6 = -4
        self.assertAlmostEqual(KNearestNeighbors.euclidean_distance(p5, p6), 2)

    def test_calculate_distance_r2(self):
        """Calculate distance for points in R^2"""
        p1 = [0, 0]
        p2 = [-2, -3]
        self.assertAlmostEqual(KNearestNeighbors.euclidean_distance(p1, p2), math.sqrt(13))

        p3 = [2, -2]
        p4 = [-2, 2]
        self.assertAlmostEqual(KNearestNeighbors.euclidean_distance(p3, p4), math.sqrt(32))

    def test_calculate_distance_r3(self):
        """Calculate distance for points in R^3"""
        p1 = [1, -1, 2]
        p2 = [-1, -3, 1]
        self.assertAlmostEqual(KNearestNeighbors.euclidean_distance(p1, p2), math.sqrt(9))

        p3 = [2, 3, 4]
        p4 = [4, 3, 2]
        self.assertAlmostEqual(KNearestNeighbors.euclidean_distance(p3, p4), math.sqrt(8))

    def test_calculate_distances_r1(self):
        """Calculate euclidean distances for array of points in R^1"""
        point = 3
        points_arr = [1, 2, 3, 4, 5, 6]

        np.testing.assert_array_almost_equal(KNearestNeighbors.calculate_distances(points_arr, point), [2, 1, 0, 1, 2, 3])
    def test_calculate_distances_r3(self):
        """Calculate euclidean distances for array of points in R^3"""
        point = [-1, -3, 1]

        p1 = [1, -1, 2]
        distance1 = math.sqrt(9)
        p2 = [0, -2, 1]
        distance2 = math.sqrt(2)
        p3 = [1, 1, 0]
        distance3 = math.sqrt(21)
        points_arr = [p1, p2, p3]
        np.testing.assert_array_almost_equal(KNearestNeighbors.calculate_distances(points_arr, point), [distance1, distance2, distance3])

    def test_nearest_points_r1(self):
        """Find the nearest points in R^1"""
        point = 3
        k = 3
        points_arr = [10, 4, 0, 2, 11, -2, 5]
        self.assertCountEqual(
            KNearestNeighbors.nearest_points(points_arr, point, k),
            [2, 4, 5]
        )

    def test_nearest_points_r1_args(self):
        """Find the nearest points in R^1"""
        point = 3
        k = 3
        points_arr = [10, 4, 0, 2, 11, -2, 5]
        self.assertCountEqual(
            KNearestNeighbors.nearest_points_args(points_arr, point, k),
            [1, 3, 6]
        )

    def test_nearest_points_r3(self):
        """Find the nearest points in R^3"""
        point = (1, 2, 3)
        k = 2
        points_arr = [[2, 1, 3],  # distance sqrt(2)
                      [3, 1, 1],  # distance sqrt(5)
                      [0, 0, 0],  # distance sqrt(14)
                      [1, 2, 2]]   # distance sqrt(1)

        self.assertCountEqual(
            KNearestNeighbors.nearest_points(points_arr, point, k).tolist(),
            [[1, 2, 2], [2, 1, 3]]
        )

    def test_nearest_points_args_r3(self):
        """Find the nearest points in R^3"""
        point = (1, 2, 3)
        k = 2
        points_arr = [[2, 1, 3],  # distance sqrt(2)
                      [3, 1, 1],  # distance sqrt(5)
                      [0, 0, 0],  # distance sqrt(14)
                      [1, 2, 2]]   # distance sqrt(1)

        self.assertCountEqual(
            KNearestNeighbors.nearest_points_args(points_arr, point, k).tolist(),
            [0, 3]
        )

    def test_predict_samples_r2(self):
        X = [[1, 2],
             [4, 5],
             [7, 8],
             [13, 10],
             [-1, 0],
             [15, 20]]
        y = [6, 3, 8, 2, 6, 2]
        k = 2
        samples = [[0, 0],
                   [5, 6],
                   [20, 30]]

        np.testing.assert_array_almost_equal(
            KNearestNeighbors.predict_samples(X, y, samples, k),
            [6, 5.5, 2]
        )

if __name__ == '__main__':
    unittest.main()
