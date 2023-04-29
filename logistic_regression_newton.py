import numpy as np
from numpy.typing import ArrayLike

class LogisticRegressionNewton:
    _beta: np.ndarray

    # calculate element H_k_l
    @staticmethod
    def calculate_hessian_element(y: np.ndarray, X: np.ndarray, beta: np.ndarray, k_idx: int, l_idx: int):
        return (1 / len(beta)) * (
                    (y * X[:, k_idx] * X[:, l_idx] * np.exp(LogisticRegressionNewton.g_func_vec(y, beta, X))) / \
                    (1 + np.exp(LogisticRegressionNewton.g_func_vec(y, beta, X)))).sum()

    @staticmethod
    def calculate_J_diff_element(y: np.ndarray, X: np.ndarray, beta: np.ndarray, k_idx: int):
        return -(1 / len(beta)) * (
            y * X[:, k_idx] * np.exp(LogisticRegressionNewton.g_func_vec(y, beta, X)) / (1 + np.exp(LogisticRegressionNewton.g_func_vec(y, beta, X)))
        ).sum()

    @staticmethod
    def calculate_J_diff(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        def inner_calculate_J_diff_element(k_idx, _):
            return LogisticRegressionNewton.calculate_J_diff_element(y, X, beta, int(k_idx))

        def generic_f(shape, inner_calculate_J_diff_element):
            fv = np.vectorize(inner_calculate_J_diff_element)
            return np.fromfunction(fv, shape)

        return generic_f((len(beta), 1), inner_calculate_J_diff_element)

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

    @staticmethod
    def updated_beta(y: np.ndarray, X: np.ndarray, beta: np.ndarray):
        return beta + np.matmul(LogisticRegressionNewton.calculate_hessian_inverse(y, X, beta),
                                LogisticRegressionNewton.calculate_J_diff(y, X, beta)).T


    def fit(self, X: ArrayLike, y: ArrayLike):
        X_np = np.insert(np.array(X), 0, 1, axis=1)
        self._beta = np.zeros(X_np.shape[1])
        y_np = np.array(y)
        for _ in range(max_epochs):
            self._beta = LogisticRegressionNewton.updated_beta(y_np, X_np, self._beta)

    @staticmethod
    def calculate_yhat_vector(beta: np.ndarray, X):
        preds = 1 / (1 + np.exp(-np.matmul(beta, X.T))).reshape(-1)
        return [1 if i>0.5 else 0 for i in preds]
    def predict(self, X: ArrayLike):
        X_np = np.insert(np.array(X), 0, 1, axis=1)
        return LogisticRegressionNewton.calculate_yhat_vector(self._beta, X_np)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
data = load_breast_cancer()
scaler = StandardScaler()
X_train = scaler.fit_transform(data.data[:420])
X_test = scaler.transform(data.data[420:])
y_train = data.target[:420]
y_test = data.target[420:]

model = LogisticRegressionNewton()
model.fit(X_train, y_train, max_epochs=10)
predictions = model.predict(X_test)

print("Score:")
print((len(y_test) - np.array(np.abs(predictions - y_test)).sum()) / len(y_test))