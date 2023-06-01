import numpy as np


class NaiveBayesClassifier:
    __X: np.ndarray
    __y: np.ndarray

    @staticmethod
    def calculate_theta_y1(y: np.ndarray) -> float:
        return y.sum() / len(y)

    @staticmethod
    def calculate_theta_y0(y: np.ndarray) -> float:
        leny = len(y)
        return (leny - y.sum()) / leny

    @staticmethod
    def calculate_theta_j_y1(X: np.ndarray, y: np.ndarray, j: int) -> float:
        X_j = X[:, j]
        return len(X_j[(y == 1) & (X_j == 1)]) / y.sum()

    @staticmethod
    def calculate_theta_j_y0(X: np.ndarray, y: np.ndarray, j: int) -> float:
        X_j = X[:, j]
        return len(X_j[(y == 1) & (X_j == 1)]) / (len(y) - y.sum())

    @staticmethod
    def calculate_product_theta_y1(X: np.ndarray, y: np.ndarray, sample: np.ndarray) -> float:
        return np.array([NaiveBayesClassifier.calculate_theta_j_y1(X, y, j) for j in np.where(sample == 1)[0]]).prod()

    @staticmethod
    def calculate_product_theta_y0(X: np.ndarray, y: np.ndarray, sample: np.ndarray) -> float:
        return np.array([NaiveBayesClassifier.calculate_theta_j_y0(X, y, j) for j in np.where(sample == 0)[0]]).prod()

    @staticmethod
    def calculate_class_1_probability(X: np.ndarray, y: np.ndarray, sample: np.ndarray) -> float:
        theta_y1 = NaiveBayesClassifier.calculate_theta_y1(y)
        theta_y0 = NaiveBayesClassifier.calculate_theta_y0(y)
        product_theta_y1 = NaiveBayesClassifier.calculate_product_theta_y1(X, y, sample)
        product_theta_y0 = NaiveBayesClassifier.calculate_product_theta_y0(X, y, sample)
        nominator = product_theta_y1 * theta_y1
        denominator = product_theta_y1 * theta_y1 + product_theta_y0 * theta_y0
        return nominator / denominator

    @staticmethod
    def calculate_class_0_probability(X: np.ndarray, y: np.ndarray, sample: np.ndarray) -> float:
        theta_y1 = NaiveBayesClassifier.calculate_theta_y1(y)
        theta_y0 = NaiveBayesClassifier.calculate_theta_y0(y)
        product_theta_y1 = NaiveBayesClassifier.calculate_product_theta_y1(X, y, sample)
        product_theta_y0 = NaiveBayesClassifier.calculate_product_theta_y0(X, y, sample)
        nominator = product_theta_y0 * theta_y0
        denominator = product_theta_y0 * theta_y0 + product_theta_y1 * theta_y1
        return nominator/denominator

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__X = X
        self.__y = y

    def predict(self, sample: np.ndarray):
        return np.argmax([NaiveBayesClassifier.calculate_class_0_probability(self.__X, self.__y, sample),
                          NaiveBayesClassifier.calculate_class_1_probability(self.__X, self.__y, sample)])

y = np.array([1, 1, 0, 1, 1, 1])

X = np.array([[1, 1, 1, 1],
              [0, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [1, 1, 0, 0],
              [1, 0, 1, 1]])

sample = np.array([0, 1, 1, 0])
# print(NaiveBayesClassifier.calculate_class_1_probability(X, y, sample))
# print(NaiveBayesClassifier.calculate_class_0_probability(X, y, sample))

nbc = NaiveBayesClassifier()
nbc.fit(X, y)
print(nbc.predict(sample))
# print(NaiveBayesClassifier.predict(X, y, sample))