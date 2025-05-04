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

# Type definitions for better typing in this file
RotationType = Units.RotationType
LengthType = Units.LengthType
Position2D = Tuple[float, float]
Position3D = Tuple[float, float, float]
RotationAngles = Tuple[float, float]




class CameraExtrinsics(Enum):
    #   {PositionName} = ((offsetX(in),offsetY(in),offsetZ(in)),(yawOffset(deg),pitchOffset(deg)))
    # Values will be added as enum members when subclassing

    @staticmethod
    def getDefaultLengthType() -> LengthType:
        return Units.LengthType.IN

    @staticmethod
    def getDefaultRotationType() -> RotationType:
        return Units.RotationType.Deg

    def getOffsetXIN(self) -> float:
        return self.value[0][0]

    def getOffsetXCM(self) -> float:
        return self.value[0][0] * 2.54

    def getOffsetYIN(self) -> float:
        return self.value[0][1]

    def getOffsetYCM(self) -> float:
        return self.value[0][1] * 2.54

    def getOffsetZIN(self) -> float:
        return self.value[0][2]

    def getOffsetZCM(self) -> float:
        return self.value[0][2] * 2.54  # Fixed typo (was using Y instead of Z)

    def getYawOffset(self) -> float:
        return self.value[1][0]

    def getPitchOffset(self) -> float:
        return self.value[1][1]

    def getYawOffsetAsRadians(self) -> float:
        return math.radians(self.value[1][0])

    def getPitchOffsetAsRadians(self) -> float:
        return math.radians(self.value[1][1])

    def get4x4AffineMatrix(
        self, lengthType: LengthType = Units.LengthType.CM
    ) -> NDArray:
        """Returns a 4x4 affine transformation matrix for the camera extrinsics"""

        x_in, y_in, z_in = self.value[0]
        yaw, pitch = map(math.radians, self.value[1])  # Convert degrees to radians

        # Handle different possible return types from convertLength
        position_result = UnitConversion.convertLength(
            (x_in, y_in, z_in), CameraExtrinsics.getDefaultLengthType(), lengthType
        )

        # Ensure we have a 3D position
        if isinstance(position_result, tuple) and len(position_result) == 3:
            x, y, z = position_result
        else:
            # Default to zeros if conversion fails
            x, y, z = 0.0, 0.0, 0.0

        # Create rotation matrix (assuming yaw around Z, pitch around Y)
        rotation_matrix = Rotation.from_euler(
            "zy", [yaw, pitch], degrees=False
        ).as_matrix()

        # Construct the 4x4 transformation matrix
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = rotation_matrix
        affine_matrix[:3, 3] = [x, y, z]  # Set translation

        return affine_matrix


class ColorCameraExtrinsics2024(CameraExtrinsics, Enum):
    #   {PositionName} = ((offsetX(in),offsetY(in),offsetZ(in)),(yawOffset(deg),pitchOffset(deg)))
    FRONTLEFT = ((13.779, 13.887, 10.744), (80, -3))
    FRONTRIGHT = ((13.779, -13.887, 10.744), (280, -3))
    REARLEFT = ((-13.116, 12.853, 10.52), (215, -3.77))
    REARRIGHT = ((-13.116, -12.853, 10.52), (145, -3.77))
    DEPTHLEFT = ((13.018, 2.548, 19.743), (24, -17))
    DEPTHRIGHT = ((13.018, -2.548, 19.743), (-24, -17))
    NONE = ((0, 0, 0), (0, 0))


class ColorCameraExtrinsics2025(CameraExtrinsics, Enum):
    #   {PositionName} = ((offsetX(in),offsetY(in),offsetZ(in)),(yawOffset(deg),pitchOffset(deg)))
    DEPTH_REAR_LEFT = ((0, 0, 0), (45.0, 10.0))
    DEPTH_REAR_RIGHT = ((0, 0, 0), (45.0, 10.0))


class ATCameraExtrinsics(CameraExtrinsics):
    def getPhotonCameraName(self):
        return self.value[2]


class ATCameraExtrinsics2024(ATCameraExtrinsics, Enum):
    #   {PositionName} = ((offsetX(in),offsetY(in),offsetZ(in)),(yawOffset(deg),pitchOffset(deg)))
    AprilTagFrontLeft = (
        (13.153, 12.972, 9.014),
        (10, -55.5),
        "Apriltag_FrontLeft_Camera",
    )
    AprilTagFrontRight = (
        (13.153, -12.972, 9.014),
        (-10, -55.5),
        "Apriltag_FrontRight_Camera",
    )
    AprilTagRearLeft = ((-13.153, 12.972, 9.014), (180, 0), "Apriltag_RearLeft_Camera")
    AprilTagRearRight = (
        (-13.153, -12.972, 9.014),
        (180, 0),
        "Apriltag_RearRight_Camera",
    )


class ATCameraExtrinsics2025(ATCameraExtrinsics, Enum):
    #   {PositionName} = ((offsetX(in),offsetY(in),offsetZ(in)),(yawOffset(deg),pitchOffset(deg)))
    AprilTagFrontLeft = ((10.14, 6.535, 6.7), (0, -21), "Apriltag_FrontLeft_Camera")
    AprilTagFrontRight = ((10.14, -6.535, 6.7), (0, -21), "Apriltag_FrontRight_Camera")
    # AprilTagBack = ((-10.25,0,7),(180,-45),"Apriltag_Back_Camera")


class CameraIntrinsics:
    def __init__(
        self,
        hres_pix: int = -1,
        vres_pix: int = -1,
        hfov_rad: float = -1,
        vfov_rad: Union[float, int] = -1,
        focal_length_mm: float = -1,
        pixel_size_mm: float = -1,
        sensor_size_mm: float = -1,
        fx_pix: float = -1,
        fy_pix: Union[int, float] = -1,
        cx_pix: Union[int, float] = -1,
        cy_pix: Union[int, float] = -1,
    ) -> None:
        self.value = (
            (hres_pix, vres_pix),  # Resolution
            (hfov_rad, vfov_rad),  # FOV
            (focal_length_mm, pixel_size_mm, sensor_size_mm),  # Physical Constants
            (fx_pix, fy_pix),  # Calibrated Fx, Fy
            (cx_pix, cy_pix),  # Calibrated Cx, Cy
        )

    """
    Create camera intrinsics at runtime.\n
    WARNING, any unfilled values may cause errors down the line. Please override default values you know you need
    """

    def getHres(self) -> float:
        return self.value[0][0]

    def getVres(self) -> float:
        return self.value[0][1]

    def getHFovRad(self) -> float:
        return self.value[1][0]

    def getVFovRad(self) -> float:
        return self.value[1][1]

    def getFocalLengthMM(self) -> float:
        return self.value[2][0]

    def getPixelSizeMM(self) -> float:
        return self.value[2][1]

    def getSensorSizeMM(self) -> float:
        return self.value[2][2]

    def getFx(self) -> float:
        assert len(self.value) > 3
        return self.value[3][0]

    def getFy(self) -> float:
        assert len(self.value) > 3
        return self.value[3][1]

    def getCx(self) -> float:
        assert len(self.value) > 4
        return self.value[4][0]

    def getCy(self) -> float:
        assert len(self.value) > 4
        return self.value[4][1]

    def __str__(self):
        return f"({self.getHres()}x{self.getVres()})-(fx:{self.getFx()}|fy:{self.getFy()}|cx:{self.getCx()}|cy:{self.getCy()})"

    @staticmethod
    def getHfov(cameraIntr: "CameraIntrinsics", radians: bool = True):
        hres = cameraIntr.getHres()
        fx = cameraIntr.getFx()

        rad = 2 * math.atan(hres / (2 * fx))
        if radians:
            return rad
        return math.degrees(rad)

    @staticmethod
    def getVfov(cameraIntr: "CameraIntrinsics", radians: bool = True):
        vres = cameraIntr.getVres()
        fy = cameraIntr.getFx()

        rad = 2 * math.atan(vres / (2 * fy))
        if radians:
            return rad
        return math.degrees(rad)

    @staticmethod
    def fromPhotonConfig(photonConfigPath):
        try:
            with open(photonConfigPath) as PV_config:
                data = json.load(PV_config)

                cameraIntrinsics = data["cameraIntrinsics"]["data"]
                fx = cameraIntrinsics[0]
                fy = cameraIntrinsics[4]
                cx = cameraIntrinsics[2]
                cy = cameraIntrinsics[5]

                width = int(data["resolution"]["width"])
                height = int(data["resolution"]["height"])

                return CameraIntrinsics(
                    hres_pix=width,
                    vres_pix=height,
                    fx_pix=fx,
                    fy_pix=fy,
                    cx_pix=cx,
                    cy_pix=cy,
                )

        except Exception as e:
            print(f"Failed to open config! {e}")
            return None

    @staticmethod
    def fromCustomConfig(customConfigPath):
        try:
            with open(customConfigPath) as custom_config:
                data = json.load(custom_config)

                cameraIntrinsics = data["CameraMatrix"]
                fx = cameraIntrinsics[0][0]
                fy = cameraIntrinsics[1][1]
                cx = cameraIntrinsics[0][2]
                cy = cameraIntrinsics[1][2]

                width = int(data["resolution"]["width"])
                height = int(data["resolution"]["height"])

                return CameraIntrinsics(
                    hres_pix=width,
                    vres_pix=height,
                    fx_pix=fx,
                    fy_pix=fy,
                    cx_pix=cx,
                    cy_pix=cy,
                )

        except Exception as e:
            print(f"Failed to open config! {e}")
            return None

    @staticmethod
    def fromCustomConfigLoaded(loadedConfig):
        try:
            data = loadedConfig

            cameraIntrinsics = data["CameraMatrix"]
            fx = cameraIntrinsics[0][0]
            fy = cameraIntrinsics[1][1]
            cx = cameraIntrinsics[0][2]
            cy = cameraIntrinsics[1][2]

            width = int(data["resolution"]["width"])
            height = int(data["resolution"]["height"])

            return CameraIntrinsics(
                hres_pix=width,
                vres_pix=height,
                fx_pix=fx,
                fy_pix=fy,
                cx_pix=cx,
                cy_pix=cy,
            )
        except Exception as e:
            print(f"Failed to create config! {e}")
            return None

    @staticmethod
    def setCapRes(cameraIntrinsics: "CameraIntrinsics", cap: cv2.VideoCapture):
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, cameraIntrinsics.getHres())
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cameraIntrinsics.getVres())
        return cap


class CameraIntrinsicsPredefined:
    #                       res             fov                     physical constants
    #   {CameraName} = ((HRes(pixels),Vres(pixels)),(Hfov(rad),Vfov(rad)),(Focal Length(mm),PixelSize(mm),sensor size(mm)), (CalibratedFx(pixels),CalibratedFy(pixels)),(CalibratedCx(pixels),CalibratedCy(pixels)))
    OV9782COLOR = CameraIntrinsics(
        640,
        480,  # Resolution
        1.22173,
        -1,  # FOV
        1.745,
        0.003,
        6.3,  # Physical Constants
        541.637,
        542.563,  # Calibrated Fx, Fy
        346.66661258567217,
        232.5032948773164,  # Calibrated Cx, Cy
    )

    SIMULATIONCOLOR = CameraIntrinsics(
        640,
        480,  # Resolution
        1.22173,
        0.9671,  # FOV
        1.745,
        0.003,
        6.3,  # Physical Constants
        604,
        414,  # Calibrated Fx, Fy
        320,
        240,  # Calibrated Cx, Cy
    )

    OAKESTIMATE = CameraIntrinsics(
        hres_pix=1920,
        vres_pix=1080,  # Resolution
        fx_pix=900,
        fy_pix=850,  # Calibrated Fx, Fy
        cx_pix=981,
        cy_pix=500,  # Calibrated Cx, Cy
    )


class OAKDLITEResolution(Enum):
    OAK4K = (3840, 2160, 30)
    OAK1080P = (1920, 1080, 60)

    @property
    def fps(self):
        return self.value[2]

    @property
    def w(self):
        return self.value[0]

    @property
    def h(self):
        return self.value[1]


class D435IResolution(Enum):
    RS720P = (1280, 720, 30)
    RS480P = (640, 480, 60)

    @property
    def fps(self):
        return self.value[2]

    @property
    def w(self):
        return self.value[0]

    @property
    def h(self):
        return self.value[1]


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






class CameraIdOffsets2024(Enum):
    # jump of 30
    FRONTLEFT = 0
    FRONTRIGHT = 30
    REARLEFT = 60
    REARRIGHT = 90
    DEPTHLEFT = 120
    DEPTHRIGHT = 150

    def getIdOffset(self) -> int:
        return self.value


class CameraIdOffsets2025(Enum):
    # jump of 30
    FRONTLEFT = 0
    FRONTRIGHT = 30
    BACK = 60
    DEPTHLEFT = 120
    DEPTHRIGHT = 150

    def getIdOffset(self) -> int:
        return self.value


class TEAM(Enum):
    RED = "red"
    BLUE = "blue"


def getCameraIfOffset2024(cameraName: str):
    for cameraIdOffset in CameraIdOffsets2024:
        if cameraIdOffset.name == cameraName:
            return cameraIdOffset

    return None


def getCameraExtrinsics2024(cameraName):
    for cameraExtrinsic in ColorCameraExtrinsics2024:
        if cameraExtrinsic.name == cameraName:
            return cameraExtrinsic

    return None


def getCameraValues2024(
    cameraName: str,
) -> tuple[CameraIntrinsics, CameraExtrinsics, CameraIdOffsets2024]:
    return (
        CameraIntrinsicsPredefined.OV9782COLOR,
        getCameraExtrinsics2024(cameraName),
        getCameraIfOffset2024(cameraName),
    )


def getCameraIfOffset2025(cameraName: str):
    for cameraIdOffset in CameraIdOffsets2025:
        if cameraIdOffset.name == cameraName:
            return cameraIdOffset

    return None


def getCameraExtrinsics2025(cameraName):
    for cameraExtrinsic in ColorCameraExtrinsics2025:
        if cameraExtrinsic.name == cameraName:
            return cameraExtrinsic

    return None


def getCameraValues2025(
    cameraName: str,
) -> tuple[CameraIntrinsics, CameraExtrinsics, CameraIdOffsets2025]:
    return (
        CameraIntrinsicsPredefined.OV9782COLOR,
        getCameraExtrinsics2025(cameraName),
        getCameraIfOffset2025(cameraName),
    )


class RealSenseSerialIDS(Enum):
    FRONTLEFTDEPTHSERIALID = "048522074864"
    FRONTRIGHTDEPTHSERIALID = "843112072752"
