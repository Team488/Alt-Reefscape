from enum import Enum

from Alt.Cameras.Parameters import CameraExtrinsics
from Alt.Core.Units import Types

class ColorCameraExtrinsics2024(Enum):
    #   {PositionName} = ((offsetX(in),offsetY(in),offsetZ(in)),(yawOffset(deg),pitchOffset(deg)))
    FRONTLEFT = CameraExtrinsics(13.779, 13.887, 10.744, 80, -3)
    FRONTRIGHT = CameraExtrinsics(13.779, -13.887, 10.744, 280, -3)
    REARLEFT = CameraExtrinsics(-13.116, 12.853, 10.52, 215, -3.77)
    REARRIGHT = CameraExtrinsics(-13.116, -12.853, 10.52, 145, -3.77)
    DEPTHLEFT = CameraExtrinsics(13.018, 2.548, 19.743, 24, -17)
    DEPTHRIGHT = CameraExtrinsics(13.018, -2.548, 19.743, -24, -17)
    NONE = CameraExtrinsics(0, 0, 0, 0, 0)


class ColorCameraExtrinsics2025(Enum):
    #   {PositionName} = ((offsetX(in),offsetY(in),offsetZ(in)),(yawOffset(deg),pitchOffset(deg)))
    DEPTH_REAR_LEFT = CameraExtrinsics(0,0,0,45.0,10.0)
    DEPTH_REAR_RIGHT = CameraExtrinsics(0,0,0,45.0,10.0)


class ATCameraExtrinsics(CameraExtrinsics):
    def __init__(self, 
        offsetX : float, offsetY : float, offsetZ : float,
        yawOffset : float, pitchOffset : float, 
        photonCameraName : str,
        translationUnits : Types.Length = Types.Length.IN,
        rotationUnits : Types.Rotation = Types.Rotation.Deg
    ):
        super().__init__(offsetX, offsetY, offsetZ, yawOffset, pitchOffset, translationUnits, rotationUnits)
        self.photonCameraName = photonCameraName

    def getPhotonCameraName(self):
        return self.photonCameraName


class ATCameraExtrinsics2024(Enum):
    AprilTagFrontLeft = ATCameraExtrinsics(
        13.153,
        12.972,
        9.014,
        10,
        -55.5,
        "Apriltag_FrontLeft_Camera",
    )
    AprilTagFrontRight = ATCameraExtrinsics(
        13.153,
        -12.972,
        9.014,
        -10,
        -55.5,
        "Apriltag_FrontRight_Camera",
    )
    AprilTagRearLeft = ATCameraExtrinsics(
        -13.153,
         12.972,
         9.014, 
         180,
         0,
         "Apriltag_RearLeft_Camera"
    )
    AprilTagRearRight = ATCameraExtrinsics(
        -13.153,
        -12.972,
        9.014,
        180,
        0,
        "Apriltag_RearRight_Camera",
    )


class ATCameraExtrinsics2025(Enum):
    AprilTagFrontLeft = ATCameraExtrinsics(
        10.14, 
        6.535, 
        6.7, 
        0, 
        -21,
        "Apriltag_FrontLeft_Camera"
    )
    
    AprilTagFrontRight = ATCameraExtrinsics(
        10.14,
        -6.535,
        6.7,
        0,
        -21, 
        "Apriltag_FrontRight_Camera"
    )
    
    AprilTagBack = ATCameraExtrinsics(
        -10.25,
        0,
        7,
        180,
        -45,
        "Apriltag_Back_Camera"
    )

