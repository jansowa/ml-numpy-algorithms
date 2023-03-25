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

if __name__ == '__main__':
    unittest.main()
