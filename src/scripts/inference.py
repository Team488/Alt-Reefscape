from Alt.Core import Neo
from Alt.ObjectLocalization.Agents.InferenceAgent import InferenceAgent
from Alt.ObjectLocalization.Inference.ModelConfig import ModelConfig

from ..Constants.CameraIntrinsics import CameraIntrinsicsPredefined
from ..Constants.ModelConfigs import ONNXMEDIUM2025
from  ..tools.load import get_asset_path

from Alt.Cameras.Captures import OpenCVCapture

if __name__ == "__main__":
    agent = InferenceAgent.bind(
        OpenCVCapture("inference", get_asset_path("reefscapevid.mp4")),
        modelConfig=ONNXMEDIUM2025,
        showFrames=True,
    )
    n = Neo()
    n.wakeAgent(agent, isMainThread=True)
    n.shutDown()
