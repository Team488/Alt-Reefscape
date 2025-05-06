from abc import abstractmethod
from enum import Enum
from functools import partial
import json
from typing import Union, Any, Dict, List, Tuple, Optional, cast
from typing import Literal, TypeVar, Generic, Type, Sequence, Callable, overload
from tools import UnitConversion, Units
import math
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray


class CommonVideos(Enum):
    ReefscapeCompilation = "assets/reefscapevid.mp4"
    Comp2024Clip = "assets/video12qual25clipped.mp4"
    ArucoCalib = "assets/arucoCalib.mp4"
    StingerCam = "assets/StingerCam.mp4"

    @property
    def path(self):
        return self.value


class SimulationEndpoints(Enum):
    FRONTRIGHTSIM = "http://localhost:3000/Robot_FrontRight%20Camera?dummy=param.mjpg"
    FRONTRIGHTAPRILTAGSIM = (
        "http://localhost:3000/Robot_FrontRight%20CameraAT?dummy=param.mjpg"
    )
    FRONTLEFTSIM = "http://localhost:3000/Robot_FrontLeft%20Camera?dummy=param.mjpg"
    REARRIGHTSIM = "http://localhost:3000/Robot_RearRight%20Camera?dummy=param.mjpg"
    REARLEFTSIM = "http://localhost:3000/Robot_RearLeft%20Camera?dummy=param.mjpg"

    @property
    def path(self):
        return self.value


class ObjectReferences(Enum):
    NOTE = (35.56, 14)  # cm , in
    BUMPERHEIGHT = (12.7, 5)  # cm, in
    ALGAEDIAMETER = (40.64, 16)  # cm in

    def getMeasurementCm(self) -> float:
        return self.value[0]

    def getMeasurementIn(self) -> float:
        return self.value[1]

class RealSenseSerialIDS(Enum):
    FRONTLEFTDEPTHSERIALID = "048522074864"
    FRONTRIGHTDEPTHSERIALID = "843112072752"
