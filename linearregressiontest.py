import unittest
import numpy as np
from linearregression import *

class TestStringMethods(unittest.TestCase):

    def test_calculate_beta(self):
        """Calculate beta for function y = 1 + 2x_1"""
        X = np.array([[3], [2]])
        y = np.array([7, 5])
        np.testing.assert_array_almost_equal(calculate_beta(X, y), [1, 2])

    def test_calculate_beta2(self):
        """Calculate beta for function y = 1 + 2x_1,
        X provided as one-dimensional array"""
        X = np.array([3, 2])
        y = np.array([7, 5])
        np.testing.assert_array_almost_equal(calculate_beta(X, y), [1, 2])

    def test_calculate_beta3(self):
        """Calculate beta for function y = 1 + x_1 + 2x_2,
        four samples"""
        X = np.array([[1, 1],
                      [0, 2],
                      [2, 1]])
        y = np.array([4, 5, 5])
        np.testing.assert_array_almost_equal(calculate_beta(X, y), [1, 1, 2])

if __name__ == '__main__':
    unittest.main()
