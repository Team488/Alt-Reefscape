from Alt.Core import Neo
from Alt.ObjectLocalization.Agents.InferenceAgent import InferenceAgent

from ..Constants.ModelConfigs import ALCOROBEST2025
from ..tools.load import get_asset_path

from Alt.Cameras.Captures import OpenCVCapture


if __name__ == "__main__":
    agent = InferenceAgent.bind(
        OpenCVCapture("inference", get_asset_path("reefscapevid.mp4")),
        modelConfig=ALCOROBEST2025,
        showFrames=True,
    )
    n = Neo()
    n.wakeAgent(agent, isMainThread=True)
    n.shutDown()
