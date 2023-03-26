import unittest
import numpy as np
from linearregression import *


class TestLinearRegression(unittest.TestCase):

    def test_calculate_beta(self):
        """Calculate beta for function y = 1 + 2x_1"""
        X = np.array([[3], [2]])
        y = np.array([7, 5])
        np.testing.assert_array_almost_equal(LinearRegression.calculate_beta(X, y), [1, 2])

    def test_calculate_beta2(self):
        """Calculate beta for function y = 1 + 2x_1,
        X provided as one-dimensional array"""
        X = np.array([3, 2])
        y = np.array([7, 5])
        np.testing.assert_array_almost_equal(LinearRegression.calculate_beta(X, y), [1, 2])

    def test_calculate_beta3(self):
        """Calculate beta for function y = 1 + x_1 + 2x_2,
        four samples"""
        X = np.array([[1, 1],
                      [0, 2],
                      [2, 1]])
        y = np.array([4, 5, 5])
        np.testing.assert_array_almost_equal(LinearRegression.calculate_beta(X, y), [1, 1, 2])

    def test_calculate_targets(self):
        """Calculate targets for given X and beta - function y = 1 + x_1 + 2x_2"""
        X = np.array([[1, 1],
                      [0, 2],
                      [2, 1]])
        beta = np.array([1, 1, 2])
        np.testing.assert_array_almost_equal(LinearRegression.calculate_targets(X, beta), [4, 5, 5])

    def test_calculate_targets2(self):
        """Calculate targets for given X and beta - function y = 1 + x_1 + 2x_2"""
        X = np.array([[1, 1],
                      [0, 2],
                      [2, 1],
                      [5, 5]])
        beta = np.array([1, 1, 2])
        np.testing.assert_array_almost_equal(LinearRegression.calculate_targets(X, beta), [4, 5, 5, 16])

    def test_fit_and_predict(self):
        """Fit model and predict targets for function y = 1 + x_1 + 2x_2"""
        X_train = np.array([[1, 1],
                            [0, 2],
                            [2, 1]])
        y = np.array([4, 5, 5])

        X_test = np.array([[2, 0],
                           [3, 1],
                           [4, 5],
                           [10, 0]])
        lr = LinearRegression()
        lr.fit(X_train, y)

        np.testing.assert_array_almost_equal(
            lr.predict(X_test),
            [3, 6, 15, 11])


if __name__ == '__main__':
    unittest.main()
