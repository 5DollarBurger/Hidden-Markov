import numpy as np
import warnings


class HMMDiscrete:
    def __init__(self, M=None, pi=[], A=[], B=[]):
        """
        This class trains a hidden Markov model with coefficients pi, A, B
        using expectation maximization based on past observed sequences using
        the Baum-Welch algorithm. Most likely sequences of underlying hidden
        states are predicted from the trained model given an observed sequence
        using the Viterbi algorithm.
        :param M: Number of hidden states
        """
        np.random.seed(seed=123)
        self.pi = pi
        self.A = A
        self.B = B
        # check consistency
        assert len(pi) == len(A), "Number of states in pi and A do not match"
        assert len(pi) == len(B), "Number of states in pi and B do not match"

        if len(pi) == 0:
            self.M = M
        else:
            self.M = len(pi)

    def _getRandomNormalized(self, shape):
        arr = np.random.random(shape)
        return arr / arr.sum(axis=1, keepdims=True)

    def _setInitialParams(self, K):
        self.pi = np.ones(self.M) / self.M
        self.A = self._getRandomNormalized(shape=(self.M, self.M))
        self.B = self._getRandomNormalized(shape=(self.M, K))

    def _setParams(self, X, max_iter=30):
        # training data characteristics
        K = max(max(seq) for seq in X) + 1
        N = len(X)

        # initial HMM parameters
        self._setInitialParams(K=K)

        costList = []
        costDelta = np.inf
        for it in range(max_iter):
            alphaList = []
            betaList = []
            scaleList = []
            logP = np.zeros(N)
            for n in range(N):
                x = X[n]
                T = len(x)
                scale = np.zeros(T)
                alpha = np.zeros(shape=(T, self.M))
                alpha[0] = self.pi * self.B[:, x[0]]
                # perform scaling
                scale[0] = alpha[0].sum()
                alpha[0] /= scale[0]
                for t in range(1, T):
                    alpha_t_prime = alpha[t-1].dot(self.A) * self.B[:, x[t]]
                    scale[t] = alpha_t_prime.sum()
                    alpha[t] = alpha_t_prime / scale[t]
                logP[n] = np.log(scale).sum()
                alphaList.append(alpha)
                scaleList.append(scale)

                beta = np.zeros(shape=(T, self.M))
                beta[-1] = 1
                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t + 1]] * beta[t + 1]) / scale[t+1]
                betaList.append(beta)

            # assert (np.all(P > 0))
            cost = np.sum(logP)
            if it > 0:
                costDelta = abs(cost - costList[-1])
            costList.append(cost)

            # now re-estimate pi, A, B
            gen = ((alphaList[n][0] * betaList[n][0]) for n in range(N))
            self.pi = sum(gen) / N

            den1 = np.zeros(shape=(self.M, 1))
            den2 = np.zeros(shape=(self.M, 1))
            a_num = np.zeros(shape=(self.M, self.M))
            b_num = np.zeros(shape=(self.M, K))
            for n in range(N):
                x = X[n]
                T = len(x)
                den1 += (alphaList[n][:-1] * betaList[n][:-1]).sum(axis=0, keepdims=True).T
                den2 += (alphaList[n] * betaList[n]).sum(axis=0, keepdims=True).T

                # numerator for A
                # a_num_n = np.zeros(shape=(self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num[i, j] += alphaList[n][t, i] * self.A[i, j] * self.B[j, x[t+1]] * betaList[n][t+1, j] \
                                           / scaleList[n][t+1]
                # a_num += a_num_n / P[n]

                # numerator for B
                # b_num_n = np.zeros(shape=(self.M, K))
                for i in range(self.M):
                    for k in range(K):
                        for t in range(T):
                            if x[t] == k:
                                b_num[i, k] += alphaList[n][t, i] * betaList[n][t, i]
                # b_num += b_num_n / P[n]
            self.A = a_num / den1
            self.B = b_num / den2

            if it > 1 and costDelta < 0.01 * abs(costList[-2] - costList[-3]):
                break

            if it == max_iter - 1:
                warnings.warn("Maximum iterations reached.")

        return costList

    def fit(self, X):
        costList = self._setParams(X)
        return costList

    def getLogLikelihood(self, X):
        """
        Calculates log likelihood of observation sequence given the model
        parameters using the forward algorithm
        :param X:
        :return:
        """
        # likelihoodList = []
        for x in X:
            T = len(x)
            scale = np.zeros(T)
            alpha = np.zeros(shape=(T, self.M))
            alpha[0] = self.pi * self.B[:, x[0]]
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0]
            for t in range(1, T):
                alpha_t_prime = alpha[t-1].dot(self.A) * self.B[:, x[t]]
                scale[t] = alpha_t_prime.sum()
                alpha[t] = alpha_t_prime / scale[t]
            # likelihood = alpha[-1].sum()
            # likelihoodList.append(likelihood)
        return sum(np.log(scale))

    def predict(self, x):
        """
        Predicts the most likely state sequence given an observed sequence x
        using the Viterbi algorithm.
        :param x: Observed sequence
        :return:
        """
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            for j in range(self.M):
                delta[t, j] = np.max(delta[t - 1] * self.A[:, j]) * self.B[j, x[t]]
                psi[t, j] = np.argmax(delta[t - 1] * self.A[:, j])

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = []
    for line in open("data/coin_data.txt"):
        # 1 for H, 0 for T
        seq = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(seq)
    ins = HMMDiscrete(M=2)
    costList = ins.fit(X=X)

    plt.plot(costList)
    plt.show()

    print("A:", ins.A)
    print("B:", ins.B)
    print("pi:", ins.pi)
    print("LL:", ins.getLogLikelihood(X=X))

    # try viterbi
    # ins.pi = np.array([0.5, 0.5])
    # ins.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    # ins.B = np.array([[0.6, 0.4], [0.3, 0.7]])
    # print("Best state sequence for: \n", np.array(X[0]))
    # print(ins.predict(x=X[0]))

