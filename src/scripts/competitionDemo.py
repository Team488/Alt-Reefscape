from Core.Agents.Partials.ReefAndObjectLocalizer import ReefAndObjectLocalizerPartial
from Captures import D435Capture
from Core.Neo import Neo
from tools.Constants import D435IResolution, ColorCameraExtrinsics2024, InferenceMode

if __name__ == "__main__":
    cap = D435Capture(D435IResolution.RS720P)
    extr = ColorCameraExtrinsics2024.NONE

    tracker = ReefAndObjectLocalizerPartial(
        capture=cap,
        cameraExtrinsics=extr,
        inferenceMode=InferenceMode.ALCOROBEST2025,
        showFrames=True,
    )

    n = Neo()

    n.wakeAgent(tracker, isMainThread=True)
    n.shutDown()
