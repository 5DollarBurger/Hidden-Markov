from src.HMMDiscrete import HMMDiscrete
import math
import os
import json

PWD = os.path.abspath(os.path.dirname(__file__))


class ShiftClassifier:
    def __init__(self, cropCal):
        self.confDict = json.load(fp=open(f"{PWD}/conf.json", "r"))
        self.cropCal = cropCal

    def _getAveStgDur(self):
        totalDur = 0
        for stgObj in self.cropCal:
            totalDur += stgObj["weeks"]
        return totalDur / len(self.cropCal)

    def _getHSLabels(self, shiftRange):
        pi = self.confDict["defaultParams"][shiftRange]["pi"]
        M = len(pi)
        numShifts = int((M - 1) / 2)
        aveStgDur = self._getAveStgDur()
        labels = tuple(aveStgDur * x for x in range(-numShifts, numShifts + 1)) #* aveStgDur
        return labels

    def _getMean(self, histDict):
        mu = 0
        for shiftWk, prob in histDict.items():
            mu += prob * shiftWk
        return mu

    def _getStd(self, histDict, mu):
        var = 0
        for shiftWk, prob in histDict.items():
            var += prob * ((shiftWk - mu) ** 2)
        return math.sqrt(var)

    def getShiftHist(self, X, shiftRange="single"):
        hmmParams = self.confDict["defaultParams"][shiftRange]

        ins = HMMDiscrete(pi=hmmParams["pi"], A=hmmParams["A"], B=hmmParams["B"])
        costList = ins.update(X=X)
        zProb = ins.filter(x=X[-1])

        histDict = {}
        shiftHSLabels = self._getHSLabels(shiftRange=shiftRange)
        for shift, prob in zip(shiftHSLabels, zProb):
            histDict[shift] = prob

        return histDict

    def getShiftDist(self, X, shiftRange="single"):
        histDict = self.getShiftHist(X=X, shiftRange=shiftRange)
        mu = self._getMean(histDict=histDict)
        sigma = self._getStd(histDict=histDict, mu=mu)
        return {"histDict": histDict, "mu": mu, "sigma": sigma}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cropCal = [
        {"stage": "seedling", "weeks": 4},
        {"stage": "tillering", "weeks": 6},
        {"stage": "panicle", "weeks": 3},
        {"stage": "flowering", "weeks": 2},
        {"stage": "filling", "weeks": 3},
        {"stage": "maturity", "weeks": 2}
    ]
    X_single = [[1, 1, 2, 2, 2, 1]]
    ins = ShiftClassifier(cropCal=cropCal)
    distDict = ins.getShiftDist(X=X_single)
    histDict = distDict["histDict"]
    # plt.bar(x=histDict.keys(), height=histDict.values())

    import scipy.stats
    import numpy as np
    aveStgDur = ins._getAveStgDur()
    x = np.linspace(-2*aveStgDur, 2*aveStgDur, 100) * 7
    y = scipy.stats.norm.pdf(x, distDict["mu"]*7, distDict["sigma"]*7)
    plt.plot(x, y, color="red")
    plt.xlabel("Shift from Calendar (days)")
    plt.ylabel("Probability Density")

    plt.show()
