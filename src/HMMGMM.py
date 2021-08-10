import numpy as np


class HMMGMM:
    def __init__(self, M, K):
        self.M = M
        self.K = K

    def _getFormattedx(self, x):
        """
        Format x into a sequence of vectors
        """
        nDim = np.array(x).ndim
        if nDim == 1:  # 1D sequence
            return [(elem,) for elem in x]
        else:
            return x

    def _getFormattedX(self, X):
        """
        Format X into a list of sequence of vectors
        """
        formattedX = []
        for x in X:
            formattedX.append(self._getFormattedx(x=x))
        return formattedX

    def _getRandomNormalized(self, shape):
        """
        Randomly intiates matrix
        """
        arr = np.random.random(shape)
        return arr / arr.sum(axis=1, keepdims=True)

    def _setInitialParams(self, X):
        N = len(X)  # number of observations
        self.pi = np.ones(self.M) / self.M
        self.A = self._getRandomNormalized(shape=(self.M, self.M))
        self.R = np.ones((self.M, self.K)) / self.K  # responsibility

        self.mu = np.zeros((self.M, self.K, self.D))
        for i in range(self.M):
            for k in range(self.K):
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                self.mu[i, k] = x[random_time_idx]

        self.sigma = np.zeros((self.M, self.K, self.D, self.D))
        for j in range(self.M):
            for k in range(self.K):
                self.sigma = np.eye(N=self.D)

    def fit(self, X, max_iter=30):
        """
        Defines HMM parameters based on training data
        """
        # X = self._getFormattedX(X=X)  #TODO: format X
        # self.L = len(X[0][0])
        self.D = X[0].shape[1]

        # initial HMM parameters
        self._setInitialParams(X=X)


if __name__ == "__main__":
    pass