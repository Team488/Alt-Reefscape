import cv2
from typing import Dict, Any, Tuple, Optional
from Core.Agents.Abstract.CameraUsingAgentBase import CameraUsingAgentBase
from Core.Agents.Abstract.PositionLocalizingAgentBase import PositionLocalizingAgentBase


class TimestampRegulatedAgentBase(CameraUsingAgentBase, PositionLocalizingAgentBase):
    """Agent -> (CameraUsingAgentBase, PositionLocalizingAgentBase) -> TimestampRegulatedAgentBase

    Using both a camera agent, and a position localizing agent, this agent adds the ability to synchronize timestamps to a give precision using a binned map. Possible todo, add a queue for a max size
    NOTE: this class should always be extended
    """

    BINSIZE: int = (
        5  # whatever the timestamp units are in, likely MS but todo figure out
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.binnedMap: Dict[int, Any] = {}

    def runPeriodic(self) -> None:
        super().runPeriodic()
        """Put binning code here TODO"""
