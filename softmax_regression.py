import numpy as np


class SoftmaxRegression:
    __class_number: int
    __theta: np.array

    def fit(self, X: np.array, y: np.array, lr: float = 0.01, max_epochs: int = 1000):
        self.__class_number = y.shape[1]
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_np = np.insert(X_temp, 0, 1, axis=1)
        self.__theta = np.ones((self.__class_number, X_np.shape[1]))
        temp_theta = np.ones((self.__class_number, X_np.shape[1]))
        for _ in range(max_epochs):
            for theta_idx in range(self.__class_number):
                temp_theta[theta_idx] = SoftmaxRegression.calculate_single_theta(X_np, y, self.__theta, lr, theta_idx)
            self.__theta = temp_theta

    def predict_proba(self, X):
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_np = np.insert(X_temp, 0, 1, axis=1)
        return [SoftmaxRegression.predict_sample(x_sample, self.__theta) for x_sample in X_np]

    def predict(self, X):
        X_temp = X if len(X.shape) > 1 else X.reshape((-1, 1))
        X_np = np.insert(X_temp, 0, 1, axis=1)
        return [np.argmax(SoftmaxRegression.predict_sample(x_sample, self.__theta)) for x_sample in X_np]

    @staticmethod
    def calculate_single_theta(X: np.array, y: np.array, theta: np.array, lr: float, class_idx: int) -> np.array:
        m = X.shape[0]
        grad = np.sum(
            [X[i] *
             (SoftmaxRegression.identity_class(
                 y[i], class_idx)
              - SoftmaxRegression.softmax(X[i], theta, class_idx))
             for i in range(m)],
            axis=0)
        return theta[class_idx] + (lr / m) * grad

    @staticmethod
    def softmax(x: np.array, theta: np.array, class_idx: int):
        class_number = theta.shape[0]
        return np.exp(theta[class_idx].T.dot(x)) / np.sum([np.exp(theta[j].T.dot(x)) for j in range(class_number)])

    @staticmethod
    def predict_sample(x: np.array, theta: np.array):
        class_number = theta.shape[0]
        return [SoftmaxRegression.softmax(x, theta, class_idx) for class_idx in range(class_number)]

    @staticmethod
    def identity_class(y: np.array, class_idx: int):
        return 1 if y.argmax() == class_idx else 0


from sklearn.datasets import load_iris

iris = load_iris()

# TODO: set whole dataset
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(iris.data)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
# TODO: set whole dataset
y_base = iris.target
y = ohe.fit_transform(iris.target.reshape(-1, 1)).toarray()

sr = SoftmaxRegression()
sr.fit(X, y, max_epochs=100)

# print(y_base)
predictions = sr.predict(X)
# print(predictions)
print(np.sum(y_base == predictions))
# print(sr.predict_proba(X))

# All thetas in single tab has the same values!
