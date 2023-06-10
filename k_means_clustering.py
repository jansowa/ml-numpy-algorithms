import numpy as np


class KMeansClustering:
    _means: np.ndarray


    @staticmethod
    def single_clustering_iteration(X: np.ndarray, old_means: np.ndarray) -> np.ndarray:
        nearest = KMeansClustering.find_nearest_mean(X, old_means)
        return np.array([np.mean(X[nearest == k], axis=0) for k in range(old_means.shape[0])])

    @staticmethod
    def find_nearest_mean_sample(X_sample: np.ndarray, means: np.ndarray):
        return np.sqrt(np.square(X_sample - means).sum(axis=1)).argmin()

    @staticmethod
    def find_nearest_mean(X: np.ndarray, means: np.ndarray) -> np.ndarray:
        return np.array([KMeansClustering.find_nearest_mean_sample(X_sample, means) for X_sample in X])

    def fit(self, X: np.ndarray, k: int) -> None:
        self._means = np.random.random_sample((k, X.shape[1]))
        old_means = np.zeros((k, X.shape[1]))

        while not np.array_equal(self._means, old_means):
            old_means = self._means
            self._means = KMeansClustering.single_clustering_iteration(X, self._means)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return KMeansClustering.find_nearest_mean(X, self._means)


X = np.array([[6, 4],
              [2, 2],
              [3, 5],
              [6, 4],
              [2, 4],
              [-6, -7],
              [-4, -6],
              [-7, -8],
              [-10, -11]])

kmc = KMeansClustering()
kmc.fit(X, 2)
print(kmc._means)
print(kmc.predict(X))
