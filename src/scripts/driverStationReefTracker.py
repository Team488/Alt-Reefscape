from Alt.Core import Neo
from Core.Agents.Abstract import ReefTrackingAgentPartial
from tools.Constants import (
    CameraIntrinsicsPredefined,
    OAKDLITEResolution,
    D435IResolution,
    CommonVideos,
    SimulationEndpoints,
)
from Captures import ConfigurableCameraCapture, OAKCapture, D435Capture, FileCapture

if __name__ == "__main":

    ReefTracker = ReefTrackingAgentPartial(
        capture=ConfigurableCameraCapture(uniqueId="1",
                                        cameraPath="assets/driverStationVideo.mp4", 
                                        cameraIntrinsics=CameraIntrinsicsPredefined.OAKESTIMATE),
        showFrames=True
    )
    """
    ReefTracker = ReefTrackingAgentPartial(
        capture=OAKCapture(OAKDLITEResolution.OAK1080P),
        showFrames=True
    )
    """
    # ReefTracker = ReefTrackingAgentPartial(cameraPath=0, cameraIntrinsics=intr, showFrames=True)

    n = Neo()
    n.wakeAgent(ReefTracker, isMainThread=True)
    n.shutDown()