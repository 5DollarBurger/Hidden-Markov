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
                alphas.append(alpha)

                P[n] = alpha[-1].sum()
                assert P[n] <= 1
                cost = np.log(P).sum()
                costList.append(cost)

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

            # now re-estimate pi, A, B
            gen = ((alphas[n][0] * betas[n][0]) / P[n] for n in range(N))
            self.pi = sum(gen) / N

            a_den = np.zeros(shape=(self.M, 1))
            a_num = 0
            r_num = np.zeros(shape=(self.M, self.K))
            r_den = np.zeros(shape=(self.M))
            mu_num = np.zeros(shape=(self.M, self.K, self.D))
            sigma_num = np.zeros(shape=(self.M, self.K, self.D, self.D))
            for n in range(N):
                x = X[n]
                T = len(x)
                B = Bs[n]

                a_den += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]

                # numerator for A
                a_num_n = np.zeros(shape=(self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T - 1):
                            a_num[i, j] += np.sum(alphas[n][t, i] * self.A[i, j] * self.B[j, x[t + 1]] * betas[n][t + 1, j])
                a_num += a_num_n / P[n]

                # numerator for B
                r_num_n = np.zeros(shape=(self.M, self.K))
                r_den_n = np.zeros(shape=(self.M))
                for j in range(self.M):
                    for k in range(self.K):
                        for t in range(T):
                            if x[t] == k:
                                r_num_n[j, k] += gamma[t, j, k]
                                r_den_n[j] += gamma[t, j, k]
                r_num += r_num_n / P[n]
                r_den += r_den_n / P[n]

                mu_num_n = np.zeros(shape=(self.M, self.K, self.D))
                sigma_num_n = np.zeros(shape=(self.M, self.K, self.D, self.D))
                for j in range(self.M):
                    for k in range(self.K):
                        for t in range(T):
                            mu_num_n += gamma[t, j, k] * x[t]
                            sigma_num_n += gamma[t, j, k] * np.outer(
                                a=x[t] - self.mu[j, k],
                                b=x[t] - self.mu[j, k]
                            )
                mu_num += mu_num_n / P[n]
                sigma_num += sigma_num_n / P[n]

            self.A = a_num / a_den
            for j in range(self.M):
                for k in range(self.K):
                    self.R[j, k] = r_num[j, k] / r_den[j]
                    self.mu[j, k] = mu_num[j, k] / r_num[j, k]
                    self.sigma[j, k] = sigma_num[j, k] / r_num[j, k]

        return costList

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

        return costList


if __name__ == "__main__":
    def simple_init():
        M = 1
        K = 1
        D = 1
        pi = np.array([1])
        A = np.array([[1]])
        R = np.array([[1]])
        mu = np.array([[[0]]])
        sigma = np.array([[[[1]]]])
        return M, K, D, pi, A, R, mu, sigma

    def big_init():
        M = 5
        K = 3
        D = 3
        pi = np.array([1, 0, 0, 0, 0])
        A = np.array([
            [0.9, 0.025, 0.025, 0.025, 0.025],
            [0.025, 0.9, 0.025, 0.025, 0.025],
            [0.025, 0.025, 0.9, 0.025, 0.025],
            [0.025, 0.025, 0.025, 0.9, 0.025],
            [0.025, 0.025, 0.025, 0.025, 0.9],
        ])
        R = np.ones(shape=(M, K)) / K
        mu = np.array([
            [[0, 0], [1, 1], [2, 2]],
            [[5, 5], [6, 6], [7, 7]],
            [[10, 10], [11, 11], [12, 12]],
            [[15, 15], [16, 16], [17, 17]],
            [[20, 20], [21, 21], [22, 22]],
        ])
        sigma = np.zeros(shape=(M, K, D, D))
        for m in range(M):
            for k in range(K):
                sigma[m, k] = np.eye(D)
        return M, K, D, pi, A, R, mu, sigma

    def get_signals(N=20, T=100, init=big_init):
        M, K, D, pi, A, R, mu, sigma = init()
        X = []
        for n in range(N):
            x = np.zeros(shape=(T, D))
            s = 0
            r = np.random.choice(a=K, p=R[s])
            x[0] = np.random.multivariate_normal(mean=mu[s][r],
                                                 cov=sigma[s][r])
            for t in range(1, T):
                s = np.random.choice(a=M, p=A[s])
                r = np.random.choice(a=K, p=A[s])
                x[t] = np.random.multivariate_normal(mean=mu[s][r],
                                                     cov=sigma[s][r])
            X.append(x)
        return X