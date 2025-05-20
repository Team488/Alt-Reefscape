import time
from typing import Optional, Any

from Alt.Cameras.Agents import CameraUsingAgentBase
from Alt.Cameras.Captures import ConfigurableCapture

from ..reefTracking.ReefPacket import ReefPacket
from ..reefTracking.reefTracker import ReefTracker


class ReefTrackingAgent(CameraUsingAgentBase):
    OBSERVATIONPOSTFIX: str = "OBSERVATIONS"
    """ Agent -> (CameraUsingAgentBase, PositionLocalizingAgentBase) -> TimestampRegulatedAgentBase -> ReefTrackingAgentBase
        This agent adds reef tracking capabilites. Must be used as partial
        If showFrames is True, you must run this agent as main
    """
    
    @classmethod
    def bind(cls, capture: ConfigurableCapture, showFrames : bool = False):
        return cls.__getBindedAgent(capture=capture, showFrames=showFrames)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tracker: Optional[ReefTracker] = None

    def create(self) -> None:
        super().create()
        self.cameraIntrinsics = self.capture.getIntrinsics()

        if self.cameraIntrinsics is None:
            raise ValueError("CameraIntrinsics not provided")

        self.tracker = ReefTracker(cameraIntrinsics=self.cameraIntrinsics)


    def runPeriodic(self) -> None:
        super().runPeriodic()

        outCoral, outAlgae, atOutput = self.tracker.getAllTracks(
            self.latestFrameMain, drawBoxes=self.showFrames or self.sendFrame
        )

        reefPkt = ReefPacket.createPacket(
            outCoral, outAlgae, "helloo", time.time() * 1000
        )
        self.updateOp.addGlobalUpdate(self.OBSERVATIONPOSTFIX, reefPkt.to_bytes())

    def getDescription(self) -> str:
        return "Gets_Reef_State"
