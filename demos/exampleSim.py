from Core.Neo import Neo
from Core.Agents.Partials.ReefAndObjectLocalizer import (
    ReefAndObjectLocalizerPartial,
)
from tools.Constants import (
    CameraIntrinsicsPredefined,
    ColorCameraExtrinsics2024,
    InferenceMode,
)
from Captures import ConfigurableCameraCapture


def startDemo() -> None:
    n = Neo()
    frameAgent = ReefAndObjectLocalizerPartial(
        capture=ConfigurableCameraCapture(
            uniqueId="SIMExample",
            cameraPath="http://localhost:3000/Robot_FrontRight%20Camera?dummy=param.mjpg",
            cameraIntrinsics=CameraIntrinsicsPredefined.SIMULATIONCOLOR,
        ),
        cameraExtrinsics=ColorCameraExtrinsics2024.FRONTRIGHT,
        inferenceMode=InferenceMode.ALCOROBEST2025,
        showFrames=True,
    )

    n.wakeAgent(frameAgent, isMainThread=True)
    n.waitForAgentsFinished()
