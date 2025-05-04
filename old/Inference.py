from enum import Enum
from typing import Any

from Alt.Core.Units import Conversions, Types

from ..Localization.DepthEstimationMethod import DepthEstimationMethod


class DefaultConfigConstants:
    confThreshold = 0.7
    drawBoxes = False
    maxDetection = None


class YoloType(Enum):
    V11 = "v11"
    V8 = "v8"
    V5 = "v5"


from enum import Enum

class Backend(Enum):
    ONNX = "onnx"
    ULTRALYTICS = "ultralytics"
    RKNN = "rknn"
    TENSORRT = "tensorrt"

available_backends = set()

try:
    from ..inference import rknnInferencer
    available_backends.add(Backend.RKNN)
except ImportError:
    pass

try:
    from ..inference import TensorrtInferencer
    available_backends.add(Backend.TENSORRT)
except ImportError:
    pass

# Always available ones
available_backends.add(Backend.ONNX)
available_backends.add(Backend.ULTRALYTICS)


class Object:
    def __init__(self, name : str, sizeCM : tuple[int, int], depthEstimationMethod : DepthEstimationMethod = None):
        self.name = name
        self.depthMethod = depthEstimationMethod
        self.sizeCM = sizeCM


class Labels:
    # name, w,h (cm)
    ROBOT = ("robot", (75, 75))
    NOTE = ("note", (35, 35))
    ALGAE = ("algae", (41, 41))
    CORAL = ("coral", (30, 12))

    @staticmethod
    def getDefaultLengthType():
        return Types.Length.CM

    def __str__(self) -> str:
        return self.value[0]

    def getSize(self, lengthType: Types.Length):
        return Conversions.convertLength(
            self.getSizeCm(), self.getDefaultLengthType(), lengthType
        )

    def getSizeCm(self):
        return self.value[1]


class ModelType(Enum):
    ALCORO = (Object.ALGAE, Object.CORAL, Object.ROBOT)
    CORO = (Object.CORAL, Object.ROBOT)
    NORO = (Object.NOTE, Object.ROBOT)


class InferenceMode(Enum):
    ONNX2024 = (
        "assets/yolov5s_fp32.onnx",
        "yolov5s-onnx-fp32",
        (Object.ROBOT, Object.NOTE),
        2024,
        Backend.ONNX,
        YoloType.V5,
        ModelType.NORO,
    )
    ONNXSMALL2025 = (
        "assets/yolov11s_fp32.onnx",
        "yolov11s-onnx-small-fp32",
        (Object.ALGAE, Object.CORAL),
        2025,
        Backend.ONNX,
        YoloType.V11,
        ModelType.CORO,
    )
    ONNXMEDIUM2025 = (
        "assets/yolov11m_fp32.onnx",
        "yolov11m-onnx-medium-fp32",
        (Object.ALGAE, Object.CORAL),
        2025,
        Backend.ONNX,
        YoloType.V11,
        ModelType.CORO,
    )
    RKNN2024FP32 = (
        "assets/yolov5s_fp32.rknn",
        "yolov5s-rknn-fp32",
        (Object.ROBOT, Object.NOTE),
        2024,
        Backend.RKNN,
        YoloType.V5,
        ModelType.NORO,
    )
    RKNN2025INT8 = (
        "assets/yolov11s_int8.rknn",
        "yolov11s-rknn-int8",
        (Object.ALGAE, Object.CORAL),
        2025,
        Backend.RKNN,
        YoloType.V11,
        ModelType.CORO,
    )

    ULTRALYTICSSMALL2025 = (
        "assets/yolov11s_fp32.pt",
        "yolov11s-pytorch-small-fp32",
        (Object.ALGAE, Object.CORAL),
        2025,
        Backend.ULTRALYTICS,
        YoloType.V11,
        ModelType.CORO,
    )
    ULTRALYTICSMED2025 = (
        "assets/yolov11m_fp32.pt",
        "yolov11s-pytorch-medium-fp32",
        (Object.ALGAE, Object.CORAL),
        2025,
        Backend.ULTRALYTICS,
        YoloType.V11,
        ModelType.CORO,
    )
    ALCOROULTRALYTICSSMALL2025BAD = (
        "assets/yolov8s_fp32_BADDD.pt",
        "verybad-yolov8s-pytorch-medium-fp32",
        (Object.ALGAE, Object.CORAL, Object.ROBOT),
        2025,
        Backend.ULTRALYTICS,
        YoloType.V8,
        ModelType.ALCORO,
    )
    ALCOROBEST2025GPUONLY = (
        "assets/yolo11sBestTensorRT.engine",
        "yolov11s-best-tensorrt",
        (Object.ALGAE, Object.CORAL, Object.ROBOT),
        2025,
        Backend.ULTRALYTICS,
        YoloType.V11,
        ModelType.ALCORO,
    )
    ALCOROBEST2025 = (
        "assets/yolo11sBest_fp32.pt",
        "yolov11s-best-pytorch",
        (Object.ALGAE, Object.CORAL, Object.ROBOT),
        2025,
        Backend.ULTRALYTICS,
        YoloType.V11,
        ModelType.ALCORO,
    )

    # TORCH todo!

    def getModelPath(self) -> str:
        return self.value[0]

    def getName(self):
        return self.value[1]

    def getLabelsAsStr(self):
        return list(map(str, self.value[2]))

    def getLabels(self):
        return self.value[2]

    def getYear(self):
        return self.value[3]

    def getBackend(self):
        return self.value[4]

    def getYoloType(self):
        return self.value[5]

    def getModelType(self):
        return self.value[6]

    @classmethod
    def getFromName(cls, name: str, default: Any = None):
        for mode in cls:
            if mode.getName() == name:
                return mode
        return default

    @classmethod
    def assertModelType(
        cls, coreInfMode: "InferenceMode", yourInfMode: "InferenceMode"
    ):
        # your model must be a subset of the core model running
        for label in yourInfMode.getLabels():
            if label not in coreInfMode.getLabels():
                return False
        return True


DEFAULTMODELTABLE = "MainProcessInferenceMODE"
DEFAULTINFERENCEMODE = InferenceMode.ALCOROBEST2025
