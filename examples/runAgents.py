from Alt.Core.Agents import Agent
from Alt.Cameras.Agents import CameraUsingAgentBase
from Alt.Cameras.Captures.OpenCVCapture import OpenCVCapture


class TestAgent(Agent):
    def create(self):
        pass

    def runPeriodic(self):
        self.Sentinel.info("Test looogog")

    def getIntervalMs(self):
        return 1000

    def isRunning(self):
        return True

    def getDescription(self):
        return "Test"

import cv2

class CamTest(CameraUsingAgentBase):
    def __init__(self, **kwargs):
        super().__init__(capture=OpenCVCapture("test",0))

    def runPeriodic(self):
        super().runPeriodic()
        cv2.putText(self.latestFrameMain, "This test will be displayed on top of the frame", (10, 20), 1, 1, (255,255,255), 1)

    def getDescription(self):
        return "test-read-webcam"


if __name__ == "__main__":
    from Alt.Core import Neo

    n = Neo()
    n.wakeAgent(TestAgent, isMainThread=False)
    n.wakeAgent(TestAgent, isMainThread=False)
    n.wakeAgent(CamTest, isMainThread=True)
    n.shutDown()
