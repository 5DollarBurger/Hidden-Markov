import numpy as np
import warnings


class HMMDNBC:
    def __init__(self, M=None, pi=[], A=[], B=[]):
        """
        This class trains a hidden Markov model with coefficients pi, A, B
        using expectation maximization based on past observed sequences using
        the Baum-Welch algorithm. Most likely sequences of underlying hidden
        states are predicted from the trained model given an observed sequence
        using the Viterbi algorithm.
        :param M: Number of hidden states
        """
        self.L = self._getNumEmissions(B=B)
        # check consistency
        self._validateParams(pi=pi, A=A, B=B)

        np.random.seed(seed=123)
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.BList = self._getFormattedEmit(B=B)

        if len(pi) == 0:
            self.M = M
        else:
            self.M = len(pi)

    def _getNumEmissions(self, B):
        """
        Get number of emission types from B
        """
        if type(B).__module__ == np.__name__:
            return max(B.ndim - 1, 1)
        else:  # B is list
            try:
                dim = len(B[0])
            except:
                return 0
            try:
                dim = len(B[0][0])
            except:  # B is emitMat
                return 1
            try:
                elem = B[0][0][0]
                # B is nested emitMat
                return len(B)
            except:  # B is emitMat
                pass

    def _validateParams(self, pi, A, B):
        assert len(pi) == len(A), "Number of states in pi and A do not match"
        if self.L > 1:  # B is nested emitMat
            for BMat in B:
                assert len(pi) == len(BMat), "Number of states in pi and B do not match"
        else:  # B is single emitMat
            numDim = np.array(B, dtype=object).ndim
            if numDim == 3:
                BMat = B[0]
            else:
                BMat = B
            assert len(pi) == len(BMat), "Number of states in pi and B do not match"
        return True

    def _validateModel(self):
        """
        Checks for consistency of HMM parameters
        """
        assert len(self.pi) != 0, "pi is empty"
        assert len(self.A) != 0, "A is empty"
        assert len(self.BList) != 0, "B is empty"
        assert self._validateParams(pi=self.pi, A=self.A, B=self.BList), "Params not consistent"
        self.L = len(self.BList)
        for l in range(self.L):
            self.BList[l] = np.array(self.BList[l])
        return True

    def _validateXAgainstModel(self, X):
        """
        Ensures formatted X observations is consistent with L
        """
        assert len(X[0][0]) == self.L, "Number of concurrent emissions not consistent with model L"
        return True

    def _validatexAgainstModel(self, x):
        """
        Ensures formatted x observations is consistent with L
        """
        assert len(x[0]) == self.L, "Number of concurrent emissions not consistent with model L"
        return True

    def _getFormattedEmit(self, B):
        BList = []
        if self.L > 1:  # B is nested emitMat
            for BMat in B:
                BList.append(np.array(BMat))
        else:  # B is single emitMat
            numDim = np.array(B).ndim
            if numDim == 3:
                BMat = np.array(B[0])
            else:
                BMat = np.array(B)
            BList.append(BMat)
        return BList

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

    def _getKListFromX(self, X):
        """
        Infers number of possible observations in sequence from training data
        """
        maxXInd = np.zeros(len(X[0][0]), dtype=np.int64)
        for seq in X:
            seqArr = np.array(seq)
            maxXInd = np.maximum(maxXInd, seqArr.max(axis=0))
        KList = maxXInd + 1
        # K = max(max(seq) for seq in X) + 1
        return KList

    def _getKListFromBList(self):
        assert len(self.BList) != 0, "Emission matrix empty"
        KList = []
        for BMat in self.BList:
            KList.append(len(BMat[0]))
        return KList

    def _getJointProbEmit(self, x, t):
        """
        Calculates joint emission probability given state across all L observations.
        https://stackoverflow.com/questions/17487356/hidden-markov-model-for-multiple-observed-variables
        """
        BProd = np.ones(shape=(self.M,))
        for l in range(self.L):
            B = self.BList[l]
            BProd *= B[:, x[t][l]]
        return BProd

    def _getLogJointProbEmit(self, x, t):
        """
        Calculates log joint emission probability given state across all L observations.
        """
        BSum = np.zeros(shape=(self.M,))
        for l in range(self.L):
            B = self.BList[l]
            BSum += np.log(B[:, x[t][l]])
        return BSum

    def _getAlpha(self, x):
        T = len(x)
        scale = np.zeros(T)
        alpha = np.zeros(shape=(T, self.M))
        alpha[0] = self.pi * self._getJointProbEmit(x=x, t=0)
        # perform scaling
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, T):
            alpha_t_prime = alpha[t-1].dot(self.A) * self._getJointProbEmit(x=x, t=t)
            scale[t] = alpha_t_prime.sum()
            alpha[t] = alpha_t_prime / scale[t]
        return alpha, scale

    def _setInitialParams(self, KList):
        self.pi = np.ones(self.M) / self.M
        self.A = self._getRandomNormalized(shape=(self.M, self.M))
        self.BList = []
        for K in KList:
            B = self._getRandomNormalized(shape=(self.M, K))
            self.BList.append(B)

    def _setParams(self, X, KList, max_iter=1000):
        # training data characteristics
        N = len(X)

        # update pi, A, B
        costList = []
        if N == 0:
            return costList
        costDelta = np.inf
        for it in range(max_iter):
            alphaList = []
            betaList = []
            scaleList = []
            logP = np.zeros(N)
            for n in range(N):
                x = np.array(X[n])
                T = len(x)
                alpha, scale = self._getAlpha(x=x)
                logP[n] = np.log(scale).sum()
                alphaList.append(alpha)
                scaleList.append(scale)

                beta = np.zeros(shape=(T, self.M))
                beta[-1] = 1
                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(self._getJointProbEmit(x=x, t=t+1) * beta[t+1]) / scale[t+1]
                betaList.append(beta)

            # assert (np.all(P > 0))
            cost = -np.sum(logP)
            if it > 0:
                costDelta = abs(cost - costList[-1])
            costList.append(cost)

            # now re-estimate pi, A, B
            gen = ((alphaList[n][0] * betaList[n][0]) for n in range(N))
            self.pi = sum(gen) / N

            den1 = np.zeros(shape=(self.M, 1))
            den2 = np.zeros(shape=(self.M, 1))
            a_num = np.zeros(shape=(self.M, self.M))
            b_numList = [np.zeros(shape=(self.M, k)) for k in KList]
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
                            BProd = self._getJointProbEmit(x=x, t=t+1)
                            a_num[i, j] += alphaList[n][t, i] * self.A[i, j] * BProd[j] * betaList[n][t+1, j] \
                                           / scaleList[n][t+1]
                # a_num += a_num_n / P[n]

                # numerator for B
                # b_num_n = np.zeros(shape=(self.M, K))
                for i in range(self.M):
                    for l in range(self.L):
                        for k in range(KList[l]):
                            for t in range(T):
                                if x[t][l] == k:
                                    b_numList[l][i, k] += alphaList[n][t, i] * betaList[n][t, i]
                # b_num += b_num_n / P[n]
            self.A = a_num / den1
            for l in range(self.L):
                self.BList[l] = b_numList[l] / den2

            if it > 1 and costDelta < 0.01 * abs(costList[-2] - costList[-3]):
                break

            if it == max_iter - 1:
                warnings.warn("Maximum iterations reached.")

        return costList

    def fit(self, X):
        """
        Defines HMM parameters based on training data
        """
        if len(X) == 0:
            return []
        X = self._getFormattedX(X=X)
        self.L = len(X[0][0])
        KList = self._getKListFromX(X)

        # initial HMM parameters
        self._setInitialParams(KList=KList)

        # update HMM parameters
        costList = self._setParams(X=X, KList=KList)
        return costList

    def update(self, X):
        """
        Updates HMM parameters based on with new training data
        """
        if len(X) == 0:
            return []
        self._validateModel()
        X = self._getFormattedX(X=X)
        self._validateXAgainstModel(X=X)
        KList = self._getKListFromBList()

        # update HMM parameters
        costList = self._setParams(X=X, KList=KList)
        return costList

    def getLogLikelihood(self, X):
        """
        Calculates log likelihood of observation sequence given the model
        parameters using the forward algorithm
        """
        self._validateModel()
        X = self._getFormattedX(X=X)
        self._validateXAgainstModel(X=X)
        logLikelihoodList = []
        for x in X:
            x = np.array(x)
            T = len(x)
            scale = np.zeros(T)
            alpha = np.zeros(shape=(T, self.M))
            alpha[0] = self.pi * self._getJointProbEmit(x=x, t=0)
            scale[0] = alpha[0].sum()
            alpha[0] /= scale[0]
            for t in range(1, T):
                alpha_t_prime = alpha[t-1].dot(self.A) * self._getJointProbEmit(x=x, t=t)
                scale[t] = alpha_t_prime.sum()
                alpha[t] = alpha_t_prime / scale[t]
            logLikelihood = np.log(scale).sum()
            logLikelihoodList.append(logLikelihood)
        return sum(logLikelihoodList)

    def predict(self, x):
        """
        Predicts the most likely state sequence given an observed sequence x
        using the Viterbi algorithm.
        :param x: Observed sequence
        """
        self._validateModel()
        x = self._getFormattedx(x=x)
        self._validatexAgainstModel(x=x)

        x = np.array(x)
        T = len(x)
        delta = np.zeros(shape=(T, self.M))
        psi = np.zeros(shape=(T, self.M))
        delta[0] = np.log(self.pi) + self._getLogJointProbEmit(x=x, t=0)
        for t in range(1, T):
            for j in range(self.M):
                delta[t, j] = np.max(delta[t-1] + np.log(self.A[:, j])) + self._getLogJointProbEmit(x=x, t=t)[j]
                psi[t, j] = np.argmax(delta[t-1] + np.log(self.A[:, j]))

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

    def filter(self, x):
        """
        Calculates the distribution of hidden states at time T
        """
        self._validateModel()
        x = self._getFormattedx(x=x)
        self._validatexAgainstModel(x=x)

        x = np.array(x)
        alpha, scale = self._getAlpha(x=x)
        stateProb = alpha[-1] / sum(alpha[-1])
        return stateProb


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X = []
    for line in open("data/coin_data.txt"):
        # 1 for H, 0 for T
        seq = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(seq)
    ins = HMMDNBC(M=2)
    costList = ins.fit(X=X)

    # ins.update(X=[X[0]])

    # plt.plot(costList)
    # plt.show()

    print("A:", ins.A)
    print("B:", ins.BList)
    print("pi:", ins.pi)
    print("LL:", ins.getLogLikelihood(X=X))

    # try viterbi
    ins.pi = np.array([0.5, 0.5])
    ins.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    ins.BList = [np.array([[0.6, 0.4], [0.3, 0.7]])]
    print("Best state sequence for: \n", np.array(X[0]))
    # print(ins.predict(x=X[0]))

    print("Terminal state distribution: \n", ins.filter(x=X[0]))

    B1 = [[0.6, 0.4], [0.3, 0.7]]
    B2 = [[0.3, 0.2, 0.5], [0.15, 0.35, 0.5]]
    BList = [B1, B2]
    ins.BList = BList
    x = [[0, 1], [0, 2]]
    print(ins.filter(x=x))

