import numpy as np


class MultinomialNaiveBayes:
    __theta_y1: float
    __theta_y0: float
    __theta_j_y1: dict
    __theta_j_y0: dict
    __vocab: np.ndarray

    @staticmethod
    def calculate_theta_y1(y: np.ndarray) -> float:
        return np.mean(y)

    @staticmethod
    def calculate_theta_y0(y: np.ndarray) -> float:
        return 1 - MultinomialNaiveBayes.calculate_theta_y1(y)

    @staticmethod
    def calculate_theta_j_y(X_specific_class: np.ndarray, vocab: np.ndarray, laplace_smoothing: float) -> dict:
        flatten = np.concatenate(X_specific_class)
        denominator = len(flatten) + laplace_smoothing * len(vocab)
        numerator = np.array([len(flatten[flatten == word]) + laplace_smoothing for word in vocab])
        values = numerator / denominator
        return dict(zip(vocab, values))

    @staticmethod
    def calculate_product_theta_y(sample: np.ndarray, theta_j: dict) -> float:
        return np.array([theta_j[j] for j in sample]).prod()

    @staticmethod
    def calculate_class_1_probability(sample: np.ndarray, theta_y1: float, theta_y0: float, theta_j_y1: dict,
                                      theta_j_y0: dict) -> float:
        product_theta_y1 = MultinomialNaiveBayes.calculate_product_theta_y(sample, theta_j_y1)
        product_theta_y0 = MultinomialNaiveBayes.calculate_product_theta_y(sample, theta_j_y0)
        nominator = product_theta_y1 * theta_y1
        denominator = product_theta_y1 * theta_y1 + product_theta_y0 * theta_y0
        return nominator / denominator

    @staticmethod
    def calculate_class_0_probability(sample: np.ndarray, theta_y1: float, theta_y0: float, theta_j_y1: dict,
                                      theta_j_y0: dict) -> float:
        product_theta_y1 = MultinomialNaiveBayes.calculate_product_theta_y(sample, theta_j_y1)
        product_theta_y0 = MultinomialNaiveBayes.calculate_product_theta_y(sample, theta_j_y0)
        nominator = product_theta_y0 * theta_y0
        denominator = product_theta_y0 * theta_y0 + product_theta_y1 * theta_y1
        return nominator / denominator

    def fit(self, X: np.ndarray, y: np.ndarray, laplace_smoothing: float = 0) -> None:
        self.__theta_y0 = MultinomialNaiveBayes.calculate_theta_y0(y)
        self.__theta_y1 = MultinomialNaiveBayes.calculate_theta_y1(y)
        self.__vocab = np.unique(np.concatenate(X))
        self.__theta_j_y0 = MultinomialNaiveBayes.calculate_theta_j_y(X[y == 0], self.__vocab, laplace_smoothing)
        self.__theta_j_y1 = MultinomialNaiveBayes.calculate_theta_j_y(X[y == 1], self.__vocab, laplace_smoothing)

    def predict(self, sample: np.ndarray):
        class_0_prob = MultinomialNaiveBayes.calculate_class_0_probability(sample, self.__theta_y1,
                                                                           self.__theta_y0, self.__theta_j_y1,
                                                                           self.__theta_j_y0)
        class_1_prob = MultinomialNaiveBayes.calculate_class_1_probability(sample, self.__theta_y1,
                                                                           self.__theta_y0, self.__theta_j_y1,
                                                                           self.__theta_j_y0)
        print([class_0_prob, class_1_prob])
        return np.argmax([class_0_prob,
                          class_1_prob])


X = np.array([[21, 23, 25, 27, 29],
              [22, 24, 26, 28, 30],
              [0, 2, 4, 6, 8, 10],
              [1, 3, 5, 7, 9],
              [21, 11, 22, 12, 23, 13, 24, 14],
              [1, 11, 2, 12, 3, 13, 4, 14],
              [15, 16, 17, 18, 19, 20, 8, 9, 10],
              [15, 16, 17, 18, 19, 20, 21, 22, 23]], dtype=np.ndarray)

y = np.array([0, 0, 1, 1, 0, 1, 1, 0])

classifier = MultinomialNaiveBayes()
classifier.fit(X, y, laplace_smoothing=1)
print(classifier.predict(np.array([0, 1, 2, 3, 4, 21, 22, 23, 24, 25])))