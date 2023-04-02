import numpy as np

class LinearRegressionOneVariableGradientDescent:

    @staticmethod
    def calculate_yhat(x, slope, intercept):
        return slope * x + intercept