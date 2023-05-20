import numpy as np
from numbers import Number

class SoftmaxRegression:
    __class_number: int
    __theta: np.array

    def fit(self, X, y):
        pass

    def train(self, X, y, lr):
        __class_number = y.shape[1]
        pass

    @staticmethod
    def softmax(x: np.array, theta:np.array, t: int, class_number: int):
        return np.exp(theta[t].T.dot(x)) / np.sum([np.exp(theta[j].T.dot(x) for j in range(class_number))])

    @staticmethod
    def identity_class(y: np.array, class_idx: int):
        return 1 if y.argmax() == class_idx else 0

y = np.array([[0, 1, 0],
              [1, 0, 0],
              [1, 0, 0],
              [0, 0, 1],
              [0, 1, 0]])

X = np.array([[0, 1],
              [2, 3],
              [4, 5],
              [6, 7],
              [8, 9]])

print(y[1].argmax())