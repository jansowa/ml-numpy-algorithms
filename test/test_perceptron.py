import unittest

from perceptron import *

class TestPerceptron(unittest.TestCase):
    def test_activation_function(self):
        X = [1, 2, 3, 4]
        weights = [5, 4, 3, 2, 1]
        self.assertAlmostEqual(Perceptron.activation_function(X, weights),
                               31)

    def step_function_test(self):
        self.assertAlmostEqual(Perceptron.step_function(0.1),
                               1)
        self.assertAlmostEqual(Perceptron.step_function(0),
                               1)
        self.assertAlmostEqual(Perceptron.step_function(-0.1),
                               0)
