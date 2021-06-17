from src.HMMDiscrete import HMMDiscrete
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
    histDict = ins.getShiftHist(X=X_single)
    plt.bar(x=histDict.keys(), height=histDict.values())
    plt.show()
