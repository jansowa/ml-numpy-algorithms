import numpy as np


class LogisticRegressionNewton:
    _beta: np.ndarray

    # calculate element H_k_l
    @staticmethod
    def calculate_hessian_element(y: np.ndarray, X: np.ndarray, beta: np.ndarray, k_idx: int, l_idx: int):
        return (1 / len(beta)) * (
                    (y * X[:, k_idx] * X[:, l_idx] * np.exp(LogisticRegressionNewton.g_func_vec(y, beta, X))) / \
                    (1 + np.exp(LogisticRegressionNewton.g_func_vec(y, beta, X)))).sum()

    @staticmethod
    def calculate_hessian(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        def inner_calculate_hessian_element(k_idx, l_idx):
            return LogisticRegressionNewton.calculate_hessian_element(y, X, beta, int(k_idx), int(l_idx))

        def generic_f(shape, inner_calculate_hessian_element):
            fv = np.vectorize(inner_calculate_hessian_element)
            return np.fromfunction(fv, shape)

        return generic_f((len(beta), len(beta)), inner_calculate_hessian_element)

    @staticmethod
    def calculate_hessian_inverse(y: np.ndarray, X: np.ndarray, beta: np.ndarray):
        return np.linalg.inv(LogisticRegressionNewton.calculate_hessian(y, X, beta))

    # -y_i * beta.T * x_i
    @staticmethod
    def g_func(y_i, beta: np.ndarray, x_i: np.ndarray):
        return -y_i * np.matmul(beta.T, x_i)

    @staticmethod
    def g_func_vec(y: np.ndarray, beta: np.ndarray, X: np.ndarray):
        return -y * np.matmul(beta, X.T)


y = np.array([1, 1, 0, 0, 1])
beta = np.array([1, 2, -3])
X = np.array([[0.3, 0.3, 0.3],
              [0.2, 0.3, 0.5],
              [0.4, 0.5, 0.6],
              [-0.2, 0.1, -0.4],
              [-0.9, 0.9, 0.5]])
print(np.matmul(LogisticRegressionNewton.calculate_hessian(y, X, beta),
                LogisticRegressionNewton.calculate_hessian_inverse(y, X, beta)))
