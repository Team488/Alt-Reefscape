from Core.Agents.Abstract import CameraUsingAgentBase
from tools.CsvParser import CsvParser, ROBOTPOSITIONKEYS
from Captures import FileCapture


class CsvSimulatedAgentBase(CameraUsingAgentBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.csvPath = kwargs.get("robotLogCsvPath", None)
        self.timeOffsetS = kwargs.get("csvAlignmentOffsetS", None)
        self.isOnRobot = kwargs.get("isOnRobot", None)

    def create(self):
        if self.isOnRobot:
            self.secPerFrame = 1 / self.capture.getFps()
            self.parser = CsvParser(
                self.csvPath, minTimestepS=self.secPerFrame, csvKeys=ROBOTPOSITIONKEYS
            )
            self.parser.removeZeroEntriesAtStart()
            self.timePassed = 0
        else:
            self.robotPose2dCMRAD = (0, 0, 0)

    def runPeriodic(self):
        super().runPeriodic()

        if self.isOnRobot:
            values = self.parser.getNearestValues(self.timePassed + self.timeOffsetS)
            # based on key order (so rotation was first when we provided csv keys...)
            rotationRad = float(values[0][1][1])
            positionXCM = int(float(values[1][1][1]) * 100)  # m -> cm
            positionYCM = int(float(values[2][1][1]) * 100)  # m -> cm
            self.robotPose2dCMRAD = (positionXCM, positionYCM, rotationRad)
        else:
            # ... not much to do here
            pass
