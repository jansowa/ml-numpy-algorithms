from unittest import TestCase
from linear_regression_one_variable_gradient_descent import *

class TestLinearRegressionOneVariableGradientDescent(TestCase):
    def test_calculate_yhat(self):
        x, slope, intercept = 1, 1, 1
        self.assertAlmostEqual(
            LinearRegressionOneVariableGradientDescent.calculate_yhat(x, slope, intercept),
            2)

        x, slope, intercept = 0, 100, 5
        self.assertAlmostEqual(
            LinearRegressionOneVariableGradientDescent.calculate_yhat(x, slope, intercept),
            5
        )

        x, slope, intercept = 5, 6, 0
        self.assertAlmostEqual(
            LinearRegressionOneVariableGradientDescent.calculate_yhat(x, slope, intercept),
            30
        )

    def test_calculate_yhat_vector(self):
        X = [1, 0, 5]
        slope = 2
        intercept = 3
        np.testing.assert_array_almost_equal(
            LinearRegressionOneVariableGradientDescent.calculate_yhat_vector(
                X, slope, intercept
            ),
            [5, 3, 13]
        )