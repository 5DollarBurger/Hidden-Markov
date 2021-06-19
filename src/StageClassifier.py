class StageClassifier:
    def __init__(self, cropCal, weekInd):
        self.cropCal = cropCal
        self.weekInd = weekInd

    # def _getRefStage(self):
    #     stgInd = 0
    #     sumWeeks = 0
    #     for stg in self.cropCal:
    #         sumWeeks += stg["weeks"]
    #         if self.weekInd >= sumWeeks:
    #             stgInd += 1
    #         else:
    #             break
    #     return stgInd

    # def _getStageProgress(self):
    #     sumDays = 0
    #     for stg in self.cropCal[:self.stgInd]:
    #         sumDays += stg["days"]
    #     daysInStg = self.dayInd + 1 - sumDays
    #     return daysInStg / self.cropCal[self.stgInd]["days"]

    def _getStgWkList(self):
        stgWkList = []
        for stgObj in self.cropCal:
            for wk in range(stgObj["weeks"]):
                stgWkList.append(stgObj["stage"])
        return stgWkList

    def getStageHist(self, weekShiftHist):
        # get list of stages across weeks
        stgWkList = self._getStgWkList()

        # initialise stage histogram dict
        stgHist = {}
        for stgObj in self.cropCal:
            stgHist[stgObj["stage"]] = 0

        # fill in stage probabilities
        maxWkInd = len(stgWkList) - 1
        for wkShift, prob in weekShiftHist.items():
            wkInd = min(max(round(self.weekInd + wkShift), 0), maxWkInd)
            stg = stgWkList[wkInd]
            stgHist[stg] += prob
        return stgHist


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json

    cropCal = [
        {"stage": "seedling", "weeks": 4},
        {"stage": "tillering", "weeks": 6},
        {"stage": "panicle", "weeks": 3},
        {"stage": "flowering", "weeks": 2},
        {"stage": "filling", "weeks": 3},
        {"stage": "maturity", "weeks": 2}
    ]

    weekShiftHist = {
        -3.3333333333333335: 0.2,
        0.0: 0.5,
        3.3333333333333335: 0.3
    }

    ins = StageClassifier(cropCal=cropCal, weekInd=8)
    print(json.dumps(ins.getStageHist(weekShiftHist=weekShiftHist), indent=4))