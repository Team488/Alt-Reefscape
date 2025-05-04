from functools import partial
from typing import Any, Tuple
from Core.Agents.Partials.ObjectLocalizingAgentBase import ObjectLocalizingAgentBase
from Core.Agents.Abstract.ReefTrackingAgentBase import ReefTrackingAgentBase
from abstract.Capture import ConfigurableCapture
from tools.Constants import CameraExtrinsics, InferenceMode, MapConstants


class ReefAndObjectLocalizer(ObjectLocalizingAgentBase, ReefTrackingAgentBase):
    """Agent -> LocalizingAgentBase -> (ObjectLocalizingAgentBase, ReefTrackingAgentBase) -> ReefAndObjectLocalizer"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # cheeky hack to set robot in the center of the field
        self.robotPose2dCMRAD: Tuple[float, float, float] = (
            MapConstants.fieldWidth.getCM() // 2,
            MapConstants.fieldHeight.getCM() // 2,
            0,
        )


def ReefAndObjectLocalizerPartial(
    capture: ConfigurableCapture,
    cameraExtrinsics: CameraExtrinsics,
    inferenceMode: InferenceMode,
    showFrames: bool = False,
) -> Any:
    """Returns a partially completed ReefAndObjectLocalizer agent. All you have to do is pass it into neo"""
    return partial(
        ReefAndObjectLocalizer,
        capture=capture,
        showFrames=showFrames,
        cameraExtrinsics=cameraExtrinsics,
        inferenceMode=inferenceMode,
    )
