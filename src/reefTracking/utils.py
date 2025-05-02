import math
import numpy as np
from tools.Units import Length, Rotation


def calculatePostHAngleDelta(
    reef_pos, robot_pos, robot_rot_rad, robotcam_offsetXY_cm, robotcam_yaw_rad
):
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
    return D_ang


def calculatePostVAngleDelta(
    reef_pos, robot_pos, robot_rot_rad, robotcam_offsetXY_cm, robotcam_yaw_rad
) -> None:
    dx = robotcam_offsetXY_cm[0]
    dy = robotcam_offsetXY_cm[1]
    cameraPos = (
        robot_pos[0] + dx * math.cos(robot_rot_rad) - dy * math.sin(robot_rot_rad),
        robot_pos[1] + dx * math.sin(robot_rot_rad) + dy * math.cos(robot_rot_rad),
    )
    obj_vec = np.subtract(reef_pos, cameraPos)
    obj_dist = np.linalg.norm(obj_vec)
