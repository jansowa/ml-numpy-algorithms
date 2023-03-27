import unittest
import numpy as np
from linearregressionclassifier import *


class TestLinearRegressionClassifier(unittest.TestCase):

    def test_fit_predict_r2(self):
        """Should fit and predict two classes in R^2"""
        X_train = [[0, 2],
                   [2, 1],
                   [2, 3],
                   [3, 5],
                   [4, 2],
                   [4, 4],
                   [6, 3],
                   [6, 5],
                   [7, 2],
                   [8, 4],
                   [8, 6],
                   [10, 5]]

        y_train = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]

        lr = LinearRegressionClassifier()
        lr.fit(X_train, y_train)

        X_test = [[6, 7],
                  [5, 1],
                  [9, 4],
                  [3, 4]]
        y_test = [0, 1, 1, 0]
        np.testing.assert_array_almost_equal(
            lr.predict(X_test),
            y_test
        )


if __name__ == '__main__':
    unittest.main()
