import time

from Alt.Cameras.Agents import CameraUsingAgentBase
from ..tools.CsvParser import CsvParser, ROBOTPOSITIONKEYS

class CsvSimulatedAgentBase(CameraUsingAgentBase):
    """ Can be used to replay a robots position from the wpilib csv logs"""
    
    def __init__(self, robotLogCsvPath : str, csvAlignmentOffsetS : float = 0, **kwargs):
        super().__init__(**kwargs)
        self.csvPath = robotLogCsvPath
        self.timeOffsetS = csvAlignmentOffsetS

    def create(self):
            self.secPerFrame = 1 / self.capture.getFps()
            self.parser = CsvParser(
                self.csvPath, minTimestepS=self.secPerFrame, csvKeys=ROBOTPOSITIONKEYS
            )
            self.parser.removeZeroEntriesAtStart()
            self.startTime = time.time()

    def runPeriodic(self):
        super().runPeriodic()

        timePassed = time.time() - self.startTime

        values = self.parser.getNearestValues(timePassed + self.timeOffsetS)
        # based on key order (so rotation was first when we provided csv keys...)
        rotationRad = float(values[0][1][1])
        positionXCM = int(float(values[1][1][1]) * 100)  # m -> cm
        positionYCM = int(float(values[2][1][1]) * 100)  # m -> cm
        self.robotPose2dCMRAD = (positionXCM, positionYCM, rotationRad)

