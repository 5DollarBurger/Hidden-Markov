class SequenceMap:
    def __init__(self, cropCal):
        self.cropCal = cropCal

    def _getStgIndWkList(self):
        stgIndWkList = []
        stgInd = 0
        for stgObj in self.cropCal:
            for wk in range(stgObj["weeks"]):
                stgIndWkList.append(stgInd)
            stgInd += 1
        return stgIndWkList

    def getSequence(self, obsStgList, shiftRange):
        stgIndWkList = self._getStgIndWkList()
        obsShiftList = []
        return obsShiftList