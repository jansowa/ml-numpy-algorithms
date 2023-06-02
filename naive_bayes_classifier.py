import numpy as np


class NaiveBayesClassifier:
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
    def calculate_theta_j_y1(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([len(X_j[(y == 1) & (X_j == 1)]) / y.sum() for X_j in X.T])

    @staticmethod
    def calculate_theta_j_y0(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([len(X_j[(y == 1) & (X_j == 1)]) / (len(y) - y.sum()) for X_j in X.T])

    @staticmethod
    def calculate_product_theta_y1(sample: np.ndarray, theta_j_y1: np.ndarray) -> float:
        return np.array([theta_j_y1[j] for j in np.where(sample == 1)[0]]).prod()

    @staticmethod
    def calculate_product_theta_y0(sample: np.ndarray, theta_j_y0: np.ndarray) -> float:
        return np.array([theta_j_y0[j] for j in np.where(sample == 0)[0]]).prod()

    @staticmethod
    def calculate_class_1_probability(sample: np.ndarray, theta_y1: float, theta_y0: float, theta_j_y1: np.ndarray,
                                      theta_j_y0: np.ndarray) -> float:
        product_theta_y1 = NaiveBayesClassifier.calculate_product_theta_y1(sample, theta_j_y1)
        product_theta_y0 = NaiveBayesClassifier.calculate_product_theta_y0(sample, theta_j_y0)
        nominator = product_theta_y1 * theta_y1
        denominator = product_theta_y1 * theta_y1 + product_theta_y0 * theta_y0
        return nominator / denominator

    @staticmethod
    def calculate_class_0_probability(sample: np.ndarray, theta_y1: float, theta_y0: float, theta_j_y1: np.ndarray,
                                      theta_j_y0: np.ndarray) -> float:
        product_theta_y1 = NaiveBayesClassifier.calculate_product_theta_y1(sample, theta_j_y1)
        product_theta_y0 = NaiveBayesClassifier.calculate_product_theta_y0(sample, theta_j_y0)
        nominator = product_theta_y0 * theta_y0
        denominator = product_theta_y0 * theta_y0 + product_theta_y1 * theta_y1
        return nominator / denominator

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__theta_y0 = NaiveBayesClassifier.calculate_theta_y0(y)
        self.__theta_y1 = NaiveBayesClassifier.calculate_theta_y1(y)
        self.__theta_j_y0 = NaiveBayesClassifier.calculate_theta_j_y0(X, y)
        self.__theta_j_y1 = NaiveBayesClassifier.calculate_theta_j_y1(X, y)

    def predict(self, sample: np.ndarray):
        class_0_prob = NaiveBayesClassifier.calculate_class_0_probability(sample, self.__theta_y1,
                                                                          self.__theta_y0, self.__theta_j_y1,
                                                                          self.__theta_j_y0)
        class_1_prob = NaiveBayesClassifier.calculate_class_1_probability(sample, self.__theta_y1,
                                                                          self.__theta_y0, self.__theta_j_y1,
                                                                          self.__theta_j_y0)
        print([class_0_prob, class_1_prob])
        return np.argmax([class_0_prob,
                          class_1_prob])


y = np.array([1, 1, 0, 1, 1, 1])

X = np.array([[1, 1, 1, 1],
              [0, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 1, 0, 0],
              [1, 1, 0, 0],
              [1, 0, 1, 1]])

sample = np.array([0, 1, 1, 0])

nbc = NaiveBayesClassifier()
nbc.fit(X, y)
print(nbc.predict(sample))
