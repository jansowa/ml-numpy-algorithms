import numpy as np


class MultinomialNaiveBayes:
    __theta_y1: float
    __theta_y0: float
    __theta_j_y1: np.ndarray
    __theta_j_y0: np.ndarray

    @staticmethod
    def calculate_theta_y1(y: np.ndarray) -> float:
        return y.sum() / len(y)

    @staticmethod
    def calculate_theta_y0(y: np.ndarray) -> float:
        leny = len(y)
        return (leny - y.sum()) / leny

    @staticmethod
    def calculate_theta_j_y0(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def calculate_theta_j_y1(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__theta_y0 = MultinomialNaiveBayes.calculate_theta_y0(y)
        self.__theta_y1 = MultinomialNaiveBayes.calculate_theta_y1(y)
        self.__theta_j_y0 = MultinomialNaiveBayes.calculate_theta_j_y0(X, y)
        self.__theta_j_y1 = MultinomialNaiveBayes.calculate_theta_j_y1(X, y)

    def predict(self, sample: np.ndarray) -> int:
        pass

X = np.array([np.array([0, 1, 0, 2]),
              np.array([0, 3, 4, 5, 1, 2, 0]),
              np.array([0, 6, 3, 6, 3, 2, 1, 6, 3, 2, 6, 1]),
              np.array([1, 0, 2]),
              np.array([0, 1, 2, 0, 2, 1]),
              np.array([6, 4, 2, 6, 1, 7]),
              np.array([8, 1, 2])],
             dtype=np.ndarray)

y = np.array([0, 0, 1, 0, 1, 1, 1])

print(np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=np.ndarray))
