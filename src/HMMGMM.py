import numpy as np
from scipy.stats import multivariate_normal as mvn


class HMMGMM:
    def __init__(self, M, K):
        self.M = M
        self.K = K  # number of Gaussians

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

    def _getAlpha(self, x, B):
        T = len(x)
        alpha = np.zeros(shape=(T, self.M))
        alpha[0] = self.pi * B[:, 0]
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * B[:, t]
        return alpha

    def _setInitialParams(self, X):
        N = len(X)  # number of observations
        self.pi = np.ones(self.M) / self.M
        self.A = self._getRandomNormalized(shape=(self.M, self.M))
        self.R = np.ones((self.M, self.K)) / self.K  # responsibility

        self.mu = np.zeros(shape=(self.M, self.K, self.D))
        for i in range(self.M):
            for k in range(self.K):
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                self.mu[i, k] = x[random_time_idx]

        self.sigma = np.zeros(shape=(self.M, self.K, self.D, self.D))
        for j in range(self.M):
            for k in range(self.K):
                self.sigma = np.eye(N=self.D)

    def _setParams(self, X, max_iter=30):
        # training data characteristics
        N = len(X)

        # update pi, A, B
        costList = []
        for it in range(max_iter):
            if it % 1 == 0:
                print("it:", it)
            alphas = []
            betas = []
            gammas = []
            Bs = []
            P = np.zeros(N)

            for n in range(N):
                x = X[n]
                T = len(x)

                B = np.zeros((self.M, T))
                component = np.zeros((self.M, self.K, T))
                for j in range(self.M):
                    for t in range(T):
                        for k in range(self.K):
                            p = self.R[j, k] * mvn.pdf(
                                x=x[t],
                                mean=self.mu[j, k],
                                sigma=self.sigma[j, k]
                            )
                            component[j, k, t] = p
                            B[j, t] += p
                Bs.append(B)

                alpha = self._getAlpha(x=x, B=B)
                P[n] = alpha[-1].sum()
                assert P[n] <= 1
                alphas.append(alpha)

                beta = np.zeros(shape=(T, self.M))
                beta[-1] = 1
                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(B[:, t + 1] * beta[t + 1])
                betas.append(beta)

                gamma = np.zeros(shape=(T, self.M, self.K))
                for t in range(T):
                    alphabeta = (alphas[n][t, :] * betas[n][t, :]).sum()
                    for j in range(self.M):
                        factor = alphas[n][t, j] * betas[n][t, j] / alphabeta
                        for k in range(self.K):
                            gamma[t, j, k] = factor * component[j, k, t] / B[j, t]
                gammas.append(gamma)

    def fit(self, X):
        """
        Defines HMM parameters based on training data
        """
        # X = self._getFormattedX(X=X)  #TODO: format X
        # self.L = len(X[0][0])
        self.D = X[0].shape[1]

        # initial HMM parameters
        self._setInitialParams(X=X)

        # update HMM parameters
        costList = self._setParams(X=X)


if __name__ == "__main__":
    pass