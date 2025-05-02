from Core.Agents.Partials.InferenceAgent import InferenceAgentPartial
from Core.Neo import Neo
from tools.Constants import InferenceMode, CameraIntrinsicsPredefined, CommonVideos
from Captures import FileCapture, ConfigurableCameraCapture

if __name__ == "__main__":
    agent = InferenceAgentPartial(
        ConfigurableCameraCapture(
            "Common_Video",
            CommonVideos.ReefscapeCompilation.path,
            CameraIntrinsicsPredefined.OV9782COLOR,
        ),
        InferenceMode.ONNXSMALL2025,
        showFrames=True,
    )
    n = Neo()
    n.wakeAgent(agent, isMainThread=False)
    n.waitForAgentsFinished()
    n.shutDown()
