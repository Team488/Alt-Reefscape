from Alt.ObjectLocalization.Inference.ModelConfig import ModelConfig
from Alt.ObjectLocalization.Constants.InferenceC import Backend, YoloType

from .Objects import ALGAE, CORAL, ROBOT
from ..tools.load import get_asset_path

# TODO move to 2024 or just delete
# ONNX2024 = ModelConfig(
#     "assets/yolov5s_fp32.onnx",
#     (Object.ROBOT, Object.NOTE),
#     2024,
#     Backend.ONNX,
#     YoloType.V5,
#     ModelType.NORO,
# )
# RKNN2024FP32 = (
#     "assets/yolov5s_fp32.rknn",
#     "yolov5s-rknn-fp32",
#     (Object.ROBOT, Object.NOTE),
#     2024,
#     Backend.RKNN,
#     YoloType.V5,
#     ModelType.NORO,
# )

ONNXSMALL2025 = ModelConfig(
    get_asset_path("yolov11s_fp32.onnx"),
    (ALGAE, CORAL),
    Backend.ONNX,
    YoloType.V11,
)
ONNXMEDIUM2025 = ModelConfig(
    get_asset_path("yolov11m_fp32.onnx"),
    (ALGAE, CORAL),
    Backend.ONNX,
    YoloType.V11,
)

RKNN2025INT8 = ModelConfig(
    get_asset_path("yolov11s_int8.rknn"),
    (ALGAE, CORAL),
    Backend.RKNN,
    YoloType.V11,
)

ULTRALYTICSSMALL2025 = ModelConfig(
    get_asset_path("yolov11s_fp32.pt"),
    (ALGAE, CORAL),
    Backend.ULTRALYTICS,
    YoloType.V11,
)
ULTRALYTICSMED2025 = ModelConfig(
    get_asset_path("yolov11m_fp32.pt"),
    (ALGAE, CORAL),
    Backend.ULTRALYTICS,
    YoloType.V11,
)
ALCOROULTRALYTICSSMALL2025BAD = ModelConfig(
    get_asset_path("yolov8s_fp32_BADDD.pt"),
    (ALGAE, CORAL, ROBOT),
    Backend.ULTRALYTICS,
    YoloType.V8,
)
ALCOROBEST2025GPUONLY = ModelConfig(
    get_asset_path("yolo11sBestTensorRT.engine"),
    (ALGAE, CORAL, ROBOT),
    Backend.ULTRALYTICS,
    YoloType.V11,
)
ALCOROBEST2025 = ModelConfig(
    get_asset_path("yolo11sBest_fp32.pt"),
    (ALGAE, CORAL, ROBOT),
    Backend.ULTRALYTICS,
    YoloType.V11,
)