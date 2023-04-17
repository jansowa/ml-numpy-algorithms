import unittest

from perceptron import *

class TestPerceptron(unittest.TestCase):
    def test_activation_function(self):
        X = [[1, 0, 1],
             [1, 1, 0],
             [0, 1, 1],
             [1, 1, 1]]
        weights = [4, 3, 2, 1]
        np.testing.assert_array_almost_equal(Perceptron.activation_function(X, weights),
                                             [8, 9, 7, 10])

    def test_step_function(self):
        np.testing.assert_array_almost_equal(
            Perceptron.step_function([0.1, -0.1, 100, -100]),
            [1, 0, 1, 0]
        )

    def test_make_prediction(self):
        X = [[-1, -1, 0],
             [-1, -1, -1],
             [1, 1, 1],
             [0, 0, 0]]
        weights = [4, 3, 2, 1]
        np.testing.assert_array_almost_equal(
            Perceptron.make_prediction(X, weights),
            [0, 0, 1, 1]
        )

    def test_fit_and_predict(self):
        X = np.array([[1, 0, 1],
                      [1, 1, 0],
                      [0, 1, 1],
                      [0, 0, 0],
                      [1, 1, 1],
                      [0, 0, 1]])
        y = [0, 1, 1, 1, 1, 0]
        p = Perceptron()
        p.fit(X, y)
        np.testing.assert_array_almost_equal(
            p.predict(X),
            y
        )
