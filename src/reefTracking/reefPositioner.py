from typing import Union
import math

import numpy as np

from Alt.Cameras.Parameters import CameraExtrinsics, MapConstants, CameraIntrinsics

# why did i make this so confusing?
# mappings from reef_idx -> april tag id
# blueAtMap = {0: 21, 1: 22, 2: 17, 3: 18, 4: 19, 5: 20}
blueAtMap = {0: 21, 1: 20, 2: 19, 3: 18, 4: 17, 5: 22}
# redAtMap = {0: 7, 1: 6, 2: 11, 3: 10, 4: 9, 5: 8}
redAtMap = {0: 7, 1: 8, 2: 9, 3: 10, 4: 11, 5: 6}


class ReefPositioner:
    def __init__(
        self,
        bluereef_center_cm=MapConstants.b_reef_center.getCM(),
        redreef_center_cm=MapConstants.r_reef_center.getCM(),
        reef_radius_cm=MapConstants.reefRadius.getCM(),
        num_post_groups=6,
    ) -> None:
        self.b_reef_center = bluereef_center_cm
        self.r_reef_center = redreef_center_cm
        self.reef_radius = reef_radius_cm
        self.anglePerPost = math.radians(360 / num_post_groups)
        self.postThresh = (math.pi / 4 + 0.05) / self.anglePerPost

    # wraps an angle between +-pi (instead of 0-2pi)
    def __wrap(self, ang):
        mod = ang % (2 * math.pi)
        if mod > math.pi:
            # wrap as negative
            return (2 * math.pi) - mod
        return mod

    def getNearedIdxs(self, final_ang) -> list[int]:
        post_idx_raw = (final_ang % (2 * math.pi)) / self.anglePerPost
        near_idxs = []

        for post_idx in range(
            math.floor(post_idx_raw - self.postThresh),
            math.ceil(post_idx_raw + self.postThresh),
        ):  # Assuming 6 post groups as per __init__
            if abs(post_idx_raw - post_idx) <= self.postThresh:
                near_idxs.append(post_idx % 6)  # cap is 5 else wraparound

        return near_idxs

    def __calculateNearestSeenPostAng(
        self,
        reef_pos,
        robot_pos,
        robot_rot_rad,
        robotcam_offsetXY_cm,
        robotcam_yaw_rad,
        robotcam_fov_rad,
    ) -> tuple[float, tuple[int]]:
        dx = robotcam_offsetXY_cm[0]
        dy = robotcam_offsetXY_cm[1]
        cameraPos = (
            robot_pos[0] + dx * math.cos(robot_rot_rad) - dy * math.sin(robot_rot_rad),
            robot_pos[1] + dx * math.sin(robot_rot_rad) + dy * math.cos(robot_rot_rad),
        )
        obj_vec = np.subtract(reef_pos, cameraPos)

        obj_ang = np.arctan2(obj_vec[1], obj_vec[0])
        cam_ang = robot_rot_rad + robotcam_yaw_rad

        D_ang = obj_ang - cam_ang

        if abs(self.__wrap(D_ang)) > robotcam_fov_rad / 2:
            # out of view
            return None

        final_ang = obj_ang + math.pi
        return final_ang, self.getNearedIdxs(final_ang)

    """
        Takes robot position and camera information, and returns a sets of coordinates representing the point of the reef seen and the closest posts
        If the camera is not in the field of view of the reef, the function returns None
        If the camera is in view, it returns the coordinate of the reef it sees, and also the closest reef post idx
    """

    def getPostCoordinates(
        self,
        isBlueReef: bool,
        robot_pos_cm: tuple[int, int],
        robot_yaw_rad: float,
        robotcam_offsetXY_cm: tuple[int, int],
        robotcam_yaw_rad: float,
        robotcam_fov_rad: float,
    ) -> tuple[float, float, float, tuple[int]]:
        reef_center = self.b_reef_center if isBlueReef else self.r_reef_center
        res = self.__calculateNearestSeenPostAng(
            reef_center,
            robot_pos_cm,
            robot_yaw_rad,
            robotcam_offsetXY_cm,
            robotcam_yaw_rad,
            robotcam_fov_rad,
        )
        if res is None:
            return None
        ang, post_idxs = res
        x = reef_center[0] + math.cos(ang) * self.reef_radius
        y = reef_center[1] + math.sin(ang) * self.reef_radius
        return x, y, ang, post_idxs

    def getPostCoordinatesWconst(
        self,
        isBlueReef: bool,
        robot_pos_cm: tuple[int, int],
        robot_yaw_rad: float,
        camera_extr: CameraExtrinsics,
        camera_intr: CameraIntrinsics,
    ) -> tuple[float, float, float, tuple[int]]:
        return self.getPostCoordinates(
            isBlueReef,
            robot_pos_cm,
            robot_yaw_rad,
            (camera_extr.getOffsetXCM(), camera_extr.getOffsetYCM()),
            camera_extr.getYawOffsetAsRadians(),
            camera_intr.getHFovRad(),
        )

    @staticmethod
    def getAprilTagId(post_idx: int, isBlueSide: bool) -> Union[int, None]:
        if post_idx < 0 or post_idx >= 6:

            return None

        if isBlueSide:
            return blueAtMap.get(post_idx)
        else:
            return redAtMap.get(post_idx)
