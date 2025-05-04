import time

import cv2
from Core.Agents.Abstract.CameraUsingAgentBase import CameraUsingAgentBase
from tools.Constants import CameraIdOffsets2024
from coreinterface.FramePacket import FramePacket
from abstract.Agent import Agent


class CalibrationController(Agent):
    """Agent -> FrameDisplayer

    Agent that will automatically ingest frames and display them\n
    NOTE: Due to openCVs nature this agent must be run in the main thread\n
    """

    # CALIBPREFIXFILTER = "photonvisionfrontright"
    CALIBPREFIXFILTER = "photonvisionfrontleft"
    # CALIBPREFIXFILTER = "Adem-GamingPc"

    def create(self) -> None:
        super().create()
        # perform agent init here (eg open camera or whatnot)
        self.updateOp.setAllGlobalUpdate(
            CameraUsingAgentBase.CALIBTOGGLEPOSTFIX, True, self.filter
        )
        self.running = True

    def filter(self, path: str):
        return path.startswith(self.CALIBPREFIXFILTER)

    def runPeriodic(self) -> None:
        super().runPeriodic()

        updates = self.updateOp.readAllGlobalUpdates(
            CameraUsingAgentBase.FRAMEPOSTFIX, self.filter
        )
        for update in updates:
            path, frameBytes = update
            framePkt = FramePacket.fromBytes(frameBytes)
            cv2.imshow(path, FramePacket.getFrame(framePkt))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.running = False

    def onClose(self) -> None:
        super().onClose()
        self.updateOp.setAllGlobalUpdate(
            CameraUsingAgentBase.CALIBTOGGLEPOSTFIX, False, self.filter
        )

        cv2.destroyAllWindows()

    def isRunning(self):
        return self.running

    def getDescription(self) -> str:
        return "Start_Calibration"
