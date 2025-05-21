from Alt.Core import Neo
from ..Agents.ReefTrackingAgent import ReefTrackingAgent
from ..Constants.CameraIntrinsics import CameraIntrinsicsPredefined
from ..Constants.Captures import OpenCvWIntrinsics
from ..tools.load import get_asset_path

if __name__ == "__main__":

    ReefTracker = ReefTrackingAgent.bind(
        capture=OpenCvWIntrinsics(
            name="1",
            capturePath=get_asset_path("reefClipped.mp4"), 
            cameraIntrinsics=CameraIntrinsicsPredefined.OAKESTIMATE
        ),
        showFrames=True
    )

    n = Neo()
    n.wakeAgent(ReefTracker, isMainThread=True)
    n.shutDown()