from enum import Enum
import socket

from ..Constants.Captures import OpenCvWIntrinsics

from ..Constants.CameraIntrinsics import CameraIntrinsicsPredefined
from .ReefTrackingAgent import ReefTrackingAgent

class CameraName(Enum):
    REARRIGHT = "photonvisionrearright"
    REARLEFT = "photonvisionrearleft"
    FRONTRIGHT = "photonvisionfrontright"
    FRONTLEFT = "photonvisionfrontleft"


def getCameraName():
    name = socket.gethostname()
    return CameraName(name)


class OrangePiAgent(ReefTrackingAgent):
    """Agent -> CameraUsingAgentBase -> ReefTrackingAgentBase -> OrangePiAgent

    Agent to be run on the orange pis"""

    def __init__(self) -> None:
        # self.device_name = getCameraName().name
        # # camera values
        # cameraIntrinsics, _, _ = getCameraValues2024(self.device_name)

        cap = OpenCvWIntrinsics(
            "Orange_Pi_COLOR",
            "/dev/color_camera",
            CameraIntrinsicsPredefined.OV9782COLOR
        )
                
        super().__init__(
            capture=cap,
            showFrames=False,
            cameraIntrinsics=CameraIntrinsicsPredefined.OV9782COLOR,
        )

    def create(self) -> None:
        super().create()
        # self.Sentinel.info(f"Camera Name: {self.device_name}")

    def getDescription(self) -> str:
        return "Ingest_Camera_Run_Ai_Model_Return_Localized_Detections_And_NowAlsoTrackReef"

    def getIntervalMs(self) -> int:
        return 0
