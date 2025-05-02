import numpy as np
from mapinternals.generalUKF import generalUKF
from robotpy_apriltag import AprilTagDetection, AprilTagPoseEstimate
from tools.Constants import ATLocations
from tools.Units import LengthType
from Core import getChildLogger
from tools import Calculator
from scipy.spatial.transform import Rotation


def transform_basis_frc_to_img(T):
    """
    Transforms an affine transformation matrix from a coordinate system where:
    - x is front, y is right, z is up
    to a coordinate system where:
    - z is front, y is down, x is left

    Args:
        T: 4x4 affine transformation matrix

    Returns:
        Transformed 4x4 affine matrix
    """
    # Coordinate transformation rotation matrix (3x3)
    R_P = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

    # Extract rotation and translation from T
    R_T = T[:3, :3]  # Original rotation
    t_T = T[:3, 3]  # Original translation

    # Transform rotation
    R_new = R_P @ R_T @ R_P.T

    # Transform translation
    t_new = R_P @ t_T

    # Construct new affine transformation matrix
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new

    return T_new


def transform_basis_img_to_frc(T):
    """
    Transforms an affine transformation matrix from a coordinate system where:
    - y is down, x is right, z is front
    to a coordinate system where:
    - y is left, x is front, z is up

    Args:
        T: 4x4 affine transformation matrix

    Returns:
        Transformed 4x4 affine matrix
    """
    # Coordinate transformation rotation matrix (3x3)
    R_P = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

    # Extract rotation and translation from T
    R_T = T[:3, :3]  # Original rotation
    t_T = T[:3, 3]  # Original translation

    # Transform rotation
    R_new = R_P @ R_T @ R_P.T

    # Transform translation
    t_new = R_P @ t_T

    # Construct new affine transformation matrix
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new

    return T_new


Sentinel = getChildLogger("Pose_Solver")


class poseSolver:
    MAXAMBIGUITY = 0.3

    def __init__(self):
        initalStateVector = np.array([0, 0, 0, 0, 0, 0])  # x,y, rot, vx,vy, vrot
        measurementDim = 3  # x,y, rot

        def stateTransition(x, dt):
            x, y, r, vx, vy, vr = x
            nx = vx * dt
            ny = vy * dt
            nr = vr * dt
            vx *= 0.99
            vy *= 0.99
            vr *= 0.99
            return np.array([nx, ny, nr, vx, vy, vr])

        def measurementFunction(x):
            return x[:3]

        self.ukf = generalUKF(
            initalStateVector,
            measurementDim,
            stateTransition,
            measurementFunction,
            timeStepS=0.05,
        )

    def getLatestPoseEstimate(
        self, atResults: list[tuple[AprilTagDetection, AprilTagPoseEstimate]]
    ) -> list[float, float, float]:
        """Ingests atlocal solved poses, and gives you x(m), y(m), yaw(rad)"""
        for detection, poseEstimate in atResults:
            ambg = poseEstimate.getAmbiguity()
            err1 = poseEstimate.error1
            err2 = poseEstimate.error2
            if ambg > self.MAXAMBIGUITY:
                Sentinel.warning(
                    f"Estimate has too much ambiguity to be used! Ambg: {ambg} Max: {self.MAXAMBIGUITY}"
                )
                continue

            og_to_tag = ATLocations.getPoseAfflineMatrix(
                detection.getId(), LengthType.M
            )
            img_og_to_tag = transform_basis_frc_to_img(og_to_tag)
            # print(f"{og_to_tag=}")

            w1 = 1 / err1
            w2 = 1 / err2
            cam_to_tag1 = poseEstimate.pose1.toMatrix() * w1
            cam_to_tag2 = poseEstimate.pose2.toMatrix() * w2

            combined_cam_to_tag = np.add(cam_to_tag1, cam_to_tag2) / (w1 + w2)
            # print(f"{combined_cam_to_tag=}")
            inv_tag_to_cam = Calculator.inverse4x4Affline(combined_cam_to_tag)

            og_cam = img_og_to_tag @ inv_tag_to_cam

            frc = transform_basis_img_to_frc(og_cam)

            rot_matr = frc[:3, :3]
            yaw, pitch, roll = Rotation.from_matrix(rot_matr).as_euler(
                "ZYX", degrees=False
            )
            XY_M = frc[:2, 3]
            print(f"{frc=}")

            measurement = np.array([XY_M[0], XY_M[1], yaw])
            self.ukf.predict_and_update(measurement)
        return self.ukf.getMeasurement()
