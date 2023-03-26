import unittest
from knearestneighborsclassifier import *
import numpy as np


class TestKNearestNeighborsClassifier(unittest.TestCase):

    def test_transform_to_class(self):
        """Test method for transforming KNN predicted mean values to one of given classes"""
        classes = np.array([3, 7])
        self.assertAlmostEqual(
            KNearestNeighborsClassifier._transform_to_class(3.5, classes),
            3
        )
        self.assertAlmostEqual(
            KNearestNeighborsClassifier._transform_to_class(4.99, classes),
            3
        )
        self.assertAlmostEqual(
            KNearestNeighborsClassifier._transform_to_class(5.01, classes),
            7
        )
        self.assertAlmostEqual(
            KNearestNeighborsClassifier._transform_to_class(7, classes),
            7
        )

    def test_fit_predict_r2(self):
        """Should fit and predict two classes in R^2"""
        X_train = [[1, 1],
                   [1, 3],
                   [1, 4],
                   [2, 3],
                   [3, 2],
                   [3, 4],
                   [4, 1],
                   [5, 1]]

        y_train = [0, 1, 0, 0, 1, 1, 0, 1]
        k = 3

        knn = KNearestNeighborsClassifier()
        knn.fit(X_train, y_train, k)

        X_test = [[1, 2],
                  [4, 2],
                  [3, 3],
                  [1, 3]]
        y_test = [0, 1, 1, 0]
        np.testing.assert_array_almost_equal(
            knn.predict(X_test),
            y_test
        )


if __name__ == '__main__':
    unittest.main()
