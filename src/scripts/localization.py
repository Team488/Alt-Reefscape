from Alt.Core import Neo
from Alt.ObjectLocalization.Agents.ObjectLocalizingStep1AgentBase import ObjectLocalizingStep1AgentBase

from ..Constants.Captures import OpenCvWIntrinsics
from ..Constants.CameraIntrinsics import CameraIntrinsicsPredefined
from ..Constants.CameraExtrinsics import ColorCameraExtrinsics2024
from ..Constants.ModelConfigs import ONNXSMALL2025
from ..tools.load import get_asset_path


if __name__ == "__main__":
    objectLocalizer = ObjectLocalizingStep1AgentBase.bind(
        OpenCvWIntrinsics("objectlocal", get_asset_path("video12qual25clipped.mp4"), CameraIntrinsicsPredefined.OV9782COLOR),
        ColorCameraExtrinsics2024.NONE,
        ONNXSMALL2025,
        showFrames=True
    )

    n = Neo()
    n.wakeAgent(objectLocalizer, isMainThread=True)
    n.shutDown()
