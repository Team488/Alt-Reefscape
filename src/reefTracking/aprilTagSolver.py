from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
from Alt.Cameras.Parameters.CameraIntrinsics import CameraIntrinsics
from Alt.Cameras.Parameters.CameraExtrinsics import CameraExtrinsics
from Alt.Core.Units.Measurements import Length

from ..Constants.AprilTags import ATLocations
from ..Constants.Landmarks import MapConstants
from ..tools import Calculator
from .reefPositioner import ReefPositioner

class AprilTagSover:
    def __init__(self, camExtr: CameraExtrinsics, camIntr: CameraIntrinsics) -> None:
        self.camExtr = camExtr
        self.camIntr = camIntr
        self.breefCM = MapConstants.b_reef_center.getCM()
        self.rreefCM = MapConstants.r_reef_center.getCM()
        self.reefPositioner = ReefPositioner(
            bluereef_center_cm=self.breefCM, redreef_center_cm=self.rreefCM
        )

    def __get4x4AfflineRobotPos(self, robotPosXYCm, robotYaw) -> np.ndarray:
        rotMatrix = Rotation.from_euler("Z", [robotYaw], degrees=False).as_matrix()
        translationMatrix = np.array([robotPosXYCm[0], robotPosXYCm[1], 0])
        r = np.eye(4)
        r[:3, :3] = rotMatrix
        r[:3, 3] = translationMatrix
        return r

    def __solveAprilTagTransform(
        self, aprilTagId, robotPose2dCmRad: tuple[float, float, float]
    ) -> np.ndarray:
        og_robot = self.__get4x4AfflineRobotPos(
            robotPose2dCmRad[:2], robotPose2dCmRad[2]
        )
        robot_cam = self.camExtr.get4x4AffineMatrix(Length.CM)
        og_cam = og_robot @ robot_cam
        og_reef = ATLocations.getPoseAfflineMatrix(aprilTagId, Length.CM)
        # diff = np.subtract(og_reef[:3,3],og_cam[:3,3])

        # start = UnitConversion.toint(og_cam[:2,3])
        # end = np.add(start,UnitConversion.toint(diff[:2]))
        cam_og = Calculator.inverse4x4Affline(og_cam)
        cam_reef = cam_og @ og_reef
        # print(f"{aprilTagId=} {Rotation.from_matrix(cam_reef[:3,:3]).as_euler('ZYX',degrees=True)}")
        # print(Rotation.from_matrix(og_reef[:3, :3]).as_euler('ZYX', degrees=True))
        # print(Rotation.from_matrix(og_cam[:3,:3]).as_euler('ZYX', degrees=True))
        # print(Rotation.from_matrix(og_robot[:3,:3]).as_euler('ZYX', degrees=True))
        # print(f"robot yaw input: {robotPose2dCmRad[2]}")

        # print(f"{aprilTagId=} {og_cam}")
        # print(f"{aprilTagId=} {cam_reef}")
        return cam_reef

    def __getBluePreference(self, robotPose2dCmRad: tuple[float, float, float]) -> bool:
        """If an april tag in the blue and red side is seen, which one should be preferred"""
        robotPosCm = robotPose2dCmRad[:2]
        bdist = np.linalg.norm(np.subtract(robotPosCm, self.breefCM))
        rdist = np.linalg.norm(np.subtract(robotPosCm, self.rreefCM))
        return bdist < rdist

    def getNearestAtPose(
        self, robotPose2dCmRad: tuple[float, float, float]
    ) -> Optional[list[tuple[np.ndarray, int]]]:
        # try both blue and red reefs. If both are seen, pick the closest one
        bluepref = self.__getBluePreference(robotPose2dCmRad)
        blueRet = self.reefPositioner.getPostCoordinatesWconst(
            isBlueReef=True,
            robot_pos_cm=robotPose2dCmRad[:2],
            robot_yaw_rad=robotPose2dCmRad[2],
            camera_extr=self.camExtr,
            camera_intr=self.camIntr,
        )
        redRet = self.reefPositioner.getPostCoordinatesWconst(
            isBlueReef=False,
            robot_pos_cm=robotPose2dCmRad[:2],
            robot_yaw_rad=robotPose2dCmRad[2],
            camera_extr=self.camExtr,
            camera_intr=self.camIntr,
        )

        if blueRet is not None and redRet is not None:
            preferredRet = blueRet if bluepref else redRet
            didPreferBlue = bluepref
        elif blueRet is not None:
            preferredRet = blueRet
            didPreferBlue = True
        else:
            preferredRet = redRet
            didPreferBlue = False

        # preferredRet could still be none at this point
        if preferredRet is None:
            return None

        _, _, _, postidxs = preferredRet
        results = []
        for postidx in postidxs:
            atID = ReefPositioner.getAprilTagId(postidx, didPreferBlue)
            cam_reef = self.__solveAprilTagTransform(atID, robotPose2dCmRad)
            results.append((cam_reef, atID))
        return results
