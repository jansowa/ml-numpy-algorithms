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
    def calculate_theta_j_y1(X_class_1: np.ndarray, vocab) -> dict:
        flatten = np.concatenate(X_class_1)
        denominator = len(flatten)
        nominator = np.array([len(flatten[flatten == word]) for word in vocab])
        values = nominator / denominator
        return dict(zip(vocab, values))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__theta_y0 = MultinomialNaiveBayes.calculate_theta_y0(y)
        self.__theta_y1 = MultinomialNaiveBayes.calculate_theta_y1(y)
        self.__vocab = np.unique(np.concatenate(X))
        self.__theta_j_y0 = MultinomialNaiveBayes.calculate_theta_j_y0(X[y==0], self.__vocab)
        self.__theta_j_y1 = MultinomialNaiveBayes.calculate_theta_j_y1(X[y==1], self.__vocab)

    def predict(self, sample: np.ndarray) -> int:
        pass

X = np.array([[0, 1, 0, 2],
              [0, 3, 4, 5, 1, 2, 0],
              [0, 6, 3, 6, 3, 2, 1, 6, 3, 2, 6, 1],
              [1, 0, 2],
              [0, 1, 2, 0, 2, 1],
              [6, 4, 2, 6, 1, 7],
              [8, 1, 2]],
             dtype=np.ndarray)

y = np.array([0, 0, 1, 0, 1, 1, 1])

flatten = np.concatenate(X[y == 0])
flatten = np.concatenate(X)
denominator = len(flatten)
vocab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
nominator = [len(flatten[flatten == word]) for word in vocab]
values = np.array(nominator) / denominator
print(nominator)
print(denominator)
print(values)
print(dict(zip(vocab, values)))
