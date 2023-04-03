import numpy as np
from numpy.typing import ArrayLike


class LinearRegressionOneVariableGradientDescent:
    _slope: float
    _intercept: float

    @staticmethod
    def calculate_yhat(x: float, slope: float, intercept: float):
        return slope * x + intercept

    @staticmethod
    def calculate_yhat_vector(X: ArrayLike, slope: float, intercept: float):
        return np.array(X) * slope + intercept

    @staticmethod
    def slope_derivative(X: ArrayLike, y: ArrayLike, slope: float, intercept: float):
        X_np = np.array(X)
        y_np = np.array(y)
        return (-2 / X_np.shape[0]) *\
            (X_np * (y_np - LinearRegressionOneVariableGradientDescent.calculate_yhat_vector(X_np, slope, intercept))).sum()

    @staticmethod
    def intercept_derivative(X: ArrayLike, y: ArrayLike, slope: float, intercept: float):
        X_np = np.array(X)
        y_np = np.array(y)
        return (-2 / X_np.shape[0]) * \
            (y_np - LinearRegressionOneVariableGradientDescent.calculate_yhat_vector(X_np, slope, intercept)).sum()

    @staticmethod
    def update_parameter(lr: float, old_parameter: float, derivative: float):
        return old_parameter - lr * derivative

    def fit(self, X, y, epochs=100, lr=0.001):
        self._slope = 0
        self._intercept = 0
        for i in range(epochs):
            self._slope = LinearRegressionOneVariableGradientDescent\
                .update_parameter(lr, self._slope,
                                  LinearRegressionOneVariableGradientDescent.slope_derivative(X, y, self._slope, self._intercept))
            self._intercept = LinearRegressionOneVariableGradientDescent\
                .update_parameter(lr, self._intercept,
                                  LinearRegressionOneVariableGradientDescent.intercept_derivative(X, y, self._slope, self._intercept))

    def predict(self, X: ArrayLike):
        X_np = np.array(X)
        return LinearRegressionOneVariableGradientDescent\
            .calculate_yhat_vector(X_np, self._slope, self._intercept)
