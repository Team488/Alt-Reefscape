from Alt.Core import Neo
from ..Agents.ReefTrackingAgent import ReefTrackingAgent
from ..Constants.CameraIntrinsics import CameraIntrinsicsPredefined
from ..Constants.Captures import OpenCvWIntrinsics
from ..tools.load import get_asset_path

if __name__ == "__main":

    ReefTracker = ReefTrackingAgent.bind(
        capture=OpenCvWIntrinsics(
            name="1",
            cameraPath=get_asset_path("driverStationVideo.mp4"), 
            cameraIntrinsics=CameraIntrinsicsPredefined.OAKESTIMATE
        ),
        showFrames=True
    )

    n = Neo()
    n.wakeAgent(ReefTracker, isMainThread=True)
    n.shutDown()