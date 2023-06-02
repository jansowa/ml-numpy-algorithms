import numpy as np


class MultinomialNaiveBayes:
    __theta_y1: float
    __theta_y0: float
    __theta_j_y1: dict
    __theta_j_y0: dict
    __vocab: np.ndarray

    @staticmethod
    def calculate_theta_y1(y: np.ndarray) -> float:
        return y.sum() / len(y)

    @staticmethod
    def calculate_theta_y0(y: np.ndarray) -> float:
        leny = len(y)
        return (leny - y.sum()) / leny

    @staticmethod
    def calculate_theta_j_y0(X_class_0: np.ndarray, vocab: np.ndarray) -> dict:
        flatten = np.concatenate(X_class_0)
        denominator = len(flatten)
        nominator = np.array([len(flatten[flatten == word]) for word in vocab])
        values = nominator / denominator
        return dict(zip(vocab, values))

    @staticmethod
    def calculate_theta_j_y1(X_class_1: np.ndarray, vocab: np.ndarray) -> dict:
        flatten = np.concatenate(X_class_1)
        denominator = len(flatten)
        nominator = np.array([len(flatten[flatten == word]) for word in vocab])
        values = nominator / denominator
        return dict(zip(vocab, values))

    @staticmethod
    def calculate_product_theta_y0(sample: np.ndarray, theta_j_y0: dict) -> float:
        return np.array([theta_j_y0[j] for j in sample]).prod()

    @staticmethod
    def calculate_product_theta_y1(sample: np.ndarray, theta_j_y1: dict) -> float:
        return np.array([theta_j_y1[j] for j in sample]).prod()

    @staticmethod
    def calculate_class_1_probability(sample: np.ndarray, theta_y1: float, theta_y0: float, theta_j_y1: dict,
                                      theta_j_y0: dict) -> float:
        product_theta_y1 = MultinomialNaiveBayes.calculate_product_theta_y1(sample, theta_j_y1)
        product_theta_y0 = MultinomialNaiveBayes.calculate_product_theta_y0(sample, theta_j_y0)
        nominator = product_theta_y1 * theta_y1
        denominator = product_theta_y1 * theta_y1 + product_theta_y0 * theta_y0
        return nominator / denominator

    @staticmethod
    def calculate_class_0_probability(sample: np.ndarray, theta_y1: float, theta_y0: float, theta_j_y1: dict,
                                      theta_j_y0: dict) -> float:
        product_theta_y1 = MultinomialNaiveBayes.calculate_product_theta_y1(sample, theta_j_y1)
        product_theta_y0 = MultinomialNaiveBayes.calculate_product_theta_y0(sample, theta_j_y0)
        nominator = product_theta_y0 * theta_y0
        denominator = product_theta_y0 * theta_y0 + product_theta_y1 * theta_y1
        return nominator / denominator

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__theta_y0 = MultinomialNaiveBayes.calculate_theta_y0(y)
        self.__theta_y1 = MultinomialNaiveBayes.calculate_theta_y1(y)
        self.__vocab = np.unique(np.concatenate(X))
        self.__theta_j_y0 = MultinomialNaiveBayes.calculate_theta_j_y0(X[y == 0], self.__vocab)
        self.__theta_j_y1 = MultinomialNaiveBayes.calculate_theta_j_y1(X[y == 1], self.__vocab)

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


X = np.array([[0, 1, 0, 2, 3, 4, 5, 6, 7, 8],
              [0, 3, 4, 5, 1, 2, 0],
              [0, 6, 3, 6, 3, 2, 1, 6, 3, 2, 6, 1, 4, 5, 6, 7, 8],
              [1, 0, 2],
              [0, 1, 2, 0, 2, 1],
              [6, 4, 2, 6, 1, 7],
              [8, 1, 2]],
             dtype=np.ndarray)

y = np.array([0, 0, 1, 0, 1, 1, 1])

classifier = MultinomialNaiveBayes()
classifier.fit(X, y)
print(classifier.predict(np.array([0, 1, 2, 3, 4, 5, 6, 7])))