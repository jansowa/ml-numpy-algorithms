from unittest import TestCase
from linearregressiononevariablegradientdescent import *

class TestLinearRegressionOneVariableGradientDescent(TestCase):
    def test_calculate_yhat(self):
        x, slope, intercept = 1, 1, 1
        self.assertAlmostEquals(
            LinearRegressionOneVariableGradientDescent.calculate_yhat(x, slope, intercept),
            2)

        x, slope, intercept = 0, 100, 5
        self.assertAlmostEquals(
            LinearRegressionOneVariableGradientDescent.calculate_yhat(x, slope, intercept),
            5
        )

        x, slope, intercept = 5, 6, 0
        self.assertAlmostEquals(
            LinearRegressionOneVariableGradientDescent.calculate_yhat(x, slope, intercept),
            30
        )