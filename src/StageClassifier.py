import math


class StageClassifier:
    def __init__(self, cropCal, dayInd):
        self.cropCal = cropCal
        self.dayInd = dayInd
        self.stgInd = self._getRefStage()

    def _getAveStgDur(self):
        totalDur = 0
        for stgObj in self.cropCal:
            totalDur += stgObj["weeks"]
        return totalDur / len(self.cropCal)

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

    def _getRefStage(self):
        stgInd = 0
        sumDays = 0
        for stg in self.cropCal:
            sumDays += stg["days"]
            if self.dayInd >= sumDays:
                stgInd += 1
            else:
                break
        return stgInd

    def _getStageProgress(self):
        sumDays = 0
        for stg in self.cropCal[:self.stgInd]:
            sumDays += stg["days"]
        daysInStg = self.dayInd + 1 - sumDays
        return daysInStg / self.cropCal[self.stgInd]["days"]

    def getStageHist(self, stgShiftHist):
        stgHist = {}
        for ind in range(len(self.cropCal)):
            stgHist[ind] = 0

        maxStgInd = len(self.cropCal) - 1
        for stgShift, prob in stgShiftHist.items():
            stg = min(max(self.stgInd + stgShift, 0), maxStgInd)
            stgHist[stg] += prob
        return stgHist


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json

    cropCal = [
        {"stage": "seedling", "days": 28},
        {"stage": "tillering", "days": 42},
        {"stage": "panicle", "days": 21},
        {"stage": "flowering", "days": 14},
        {"stage": "filling", "days": 21},
        {"stage": "maturity", "days": 14}
    ]

    stgShiftHist = {
        -1: 0.08438500793171773,
        0: 0.9156149920682819,
        1: 2.3559965023904594e-16
    }

    ins = StageClassifier(cropCal=cropCal, dayInd=0)
    print(json.dumps(ins.getStageHist(stgShiftHist=stgShiftHist), indent=4))