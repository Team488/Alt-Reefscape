"""
https://robotpy.readthedocs.io/projects/apriltag/en/latest/robotpy_apriltag.html
https://pypi.org/project/photonlibpy/
PORT TO PHOTONVISION LATER: https://docs.photonvision.org/en/latest/docs/programming/photonlib/index.html

Steps:
1. Get the pose of AT
2. Find the 3D offsets of the AT -> reef branches (have a cad model for that)
3. Transform locations of reef branches via depth data -> camera frame
4. Detect if objects are "fixed" onto those branches

Measurement in inches
"""
import cv2
import numpy as np
from robotpy_apriltag import (
    AprilTagField,
    AprilTagFieldLayout,
    AprilTagDetector,
    AprilTagPoseEstimator,
)

from wpimath.geometry import Transform3d
import json

from reefTracking.aprilTagHelper import AprilTagLocal
from scipy.spatial.transform import Rotation


def affine_matrix_from_quaternion_translation(quat, translation):
    """
    Create a 4x4 affine transformation matrix from a quaternion and translation vector.

    :param quat: Quaternion (w, x, y, z)
    :param translation: Translation vector (tx, ty, tz)
    :return: 4x4 affine transformation matrix
    """
    # Convert quaternion to rotation matrix
    r = Rotation.from_quat(
        [quat[1], quat[2], quat[3], quat[0]]
    )  # (x, y, z, w) format in scipy
    rot_matrix = r.as_matrix()

    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = translation

    return T


### CAD Offset:
### X = Downwards -> Right
### Y = Upwards -> Right
### Z = Upwards

# Measurement is in Inches

# CAD to tip of the rod. (MAX Distance)
cad_to_branch_offset = {
    "L2-L": np.array([-6.756, -19.707, 2.608]),
    "L2-R": np.array([6.754, -19.707, 2.563]),
    "L3-L": np.array([-6.639, -35.606, 2.628]),
    "L3-R": np.array([6.637, -35.606, 2.583]),
    "L4-L": np.array([-6.470, -58.4175, 0.921]),  # NOT MODIFIED
    "L4-R": np.array([6.468, -58.4175, 0.876]),  # NOT MODIFIED
}


# CAD to Center of Coral
""""
cad_to_branch_offset = {
    "L2-L" : np.array([-6.470, -12.854, 9.00]),
    "L2-R" : np.array([6.468, -12.833, 9.00]),
    "L3-L" : np.array([-6.470, -23.503, 16.457]),
    "L3-R" : np.array([6.468, -23.482, 16.442]),
    "L4-L" : np.array([-6.470, -58.4175, 0.921]),
    "L4-R" : np.array([6.468, -58.4175, 0.876])
}
"""
### Camera AT Coordinate System:
#   X is LEFT -> Right  [-inf, inf]
#   Y is TOP -> Down    [-inf, inf]
#   Z is DEPTH AWAY     [0, inf]


# Convert to meters:
for branch, offset in cad_to_branch_offset.items():
    for i in range(len(offset)):
        offset[i] *= 0.0254

zoffset = 3  # in
widthOffset = 6  # in                                                                   #               ↖  (z)
heightOffset = 12  # in                                                                 #                (3)
heightClearence = 1  #                  \
reefBoxOffsets = [  # in to m                                                       #                   \
    np.array(
        [widthOffset / 2 * 0.0254, -heightClearence * 0.0254, 0]
    ),  # (1)  (x) <(1)------o
    np.array(
        [-widthOffset / 2 * 0.0254, heightOffset * 0.0254, 0]
    ),  # (2)                |
    np.array(
        [-widthOffset / 2 * 0.0254, -heightClearence * 0.0254, zoffset * 0.0254]
    ),  # (3)               (2)
]  #                    ↓ (y)
#
# all corners of a imaginary box around a point on the reef


class ReefPixelEstimator:
    def __init__(self, config_file="assets/config/640x480v2.json") -> None:
        self.helper = AprilTagLocal(config_file)
        self.loadConfig(config_file)

    def loadConfig(self, config_file) -> None:
        try:
            with open(config_file) as PV_config:
                data = json.load(PV_config)

                self.cameraIntrinsics = data["cameraIntrinsics"]["data"]
                self.fx = self.cameraIntrinsics[0]
                self.fy = self.cameraIntrinsics[4]
                self.cx = self.cameraIntrinsics[2]
                self.cy = self.cameraIntrinsics[5]

                self.K = np.array(
                    [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
                    dtype=np.float32,
                )

                self.width = int(data["resolution"]["width"])
                self.height = int(data["resolution"]["height"])

                distCoeffsSize = int(data["distCoeffs"]["cols"])
                self.distCoeffs = np.array(
                    data["distCoeffs"]["data"][0:distCoeffsSize], dtype=np.float32
                )
        except Exception as e:
            print(f"Failed to open config! {e}")

    def getReefCoordinates(self, image, drawCoordinates=True):
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        outputs = self.helper.getDetections(grayscale_image)
        orthogonalEsts = self.helper.getOrthogonalEstimates(outputs)
        coordinates = {}
        for tag_pose_estimation_orthogonal, output in zip(orthogonalEsts, outputs):
            print("ID", output.getId())

            if drawCoordinates:
                # Retrieve the corners of the AT detection
                points = []
                for corner in range(0, 4):
                    x = output.getCorner(corner).x
                    y = output.getCorner(corner).y
                    points.append([x, y])
                points = np.array(points, dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(
                    image, [points], isClosed=True, color=(0, 255, 255), thickness=3
                )

                # Retrieve the center of the AT detection
                centerX = output.getCenter().x
                centerY = output.getCenter().y
                cv2.circle(
                    image,
                    (int(centerX), int(centerY)),
                    2,
                    color=(0, 255, 255),
                    thickness=3,
                )

            tag_pose_estimation_orthogonal_pose1_matrix = (
                tag_pose_estimation_orthogonal.pose1
            )
            tag_pose_estimation_orthogonal_pose2_matrix = (
                tag_pose_estimation_orthogonal.pose2
            )
            # print(f"regular: x: {tag_pose_estimation.x}, y: {tag_pose_estimation.y}, z: {tag_pose_estimation.z}")
            print(
                f"orthogonal pose 1: x: {tag_pose_estimation_orthogonal_pose1_matrix.x}, y: {tag_pose_estimation_orthogonal_pose1_matrix.y}, z: {tag_pose_estimation_orthogonal_pose1_matrix.z}"
            )
            # print(f"orthogonal pose 2: x: {tag_pose_estimation_orthogonal_pose2_matrix.x}, y: {tag_pose_estimation_orthogonal_pose2_matrix.y}, z: {tag_pose_estimation_orthogonal_pose2_matrix.z}")
            # print("===============")

            onScreenBranches = {}
            for offset_idx, offset_3d in cad_to_branch_offset.items():
                # solve camera -> branch via camera -> tag and tag -> branch transformations
                tag_to_reef_homography = np.append(
                    offset_3d, 1.0
                )  # ensures shape is 4x4
                # camera_to_reef = np.dot(tag_pose_estimation_matrix, tag_to_reef_homography)

                camera_to_reef = np.dot(
                    tag_pose_estimation_orthogonal_pose1_matrix.toMatrix(),
                    tag_to_reef_homography,
                )

                x_cam, y_cam, z_cam, _ = camera_to_reef

                corners = []
                for boxOffset in reefBoxOffsets:
                    box_offset_homogeneous = np.append(boxOffset, 0)  # Shape: (4,)
                    tag_to_reef_corner_homography = (
                        tag_to_reef_homography + box_offset_homogeneous
                    )
                    print(f"{tag_to_reef_homography=} {tag_to_reef_corner_homography=}")
                    print(tag_to_reef_corner_homography)

                    camera_to_reef_corner = np.dot(
                        tag_pose_estimation_orthogonal_pose1_matrix.toMatrix(),
                        tag_to_reef_corner_homography,
                    )
                    corners.append(camera_to_reef_corner)

                    x_cam, y_cam, z_cam, _ = camera_to_reef_corner
                    print(
                        f"Corner 3D coords (camera frame): x={x_cam}, y={y_cam}, z={z_cam}"
                    )
                    x_cam, y_cam, z_cam, _ = camera_to_reef
                    print(
                        f"Reef post 3D coords (camera frame): x={x_cam}, y={y_cam}, z={z_cam}"
                    )

                # exit(0)

                # project the 3D point to 2D image coordinates:
                u = (self.fx * x_cam / z_cam) + self.cx
                v = (self.fy * y_cam / z_cam) + self.cy

                # project the 3d box corners to 2d image coords
                imageCorners = []
                for corner in corners:
                    x_cam, y_cam, z_cam, _ = corner
                    uC = (self.fx * x_cam / z_cam) + self.cx
                    uV = (self.fy * y_cam / z_cam) + self.cy
                    imageCorners.append((uC, uV))

                if drawCoordinates:
                    cv2.circle(image, (int(u), int(v)), 5, (0, 255, 255), 2)
                    min_x, min_y = np.min(imageCorners, axis=0)
                    max_x, max_y = np.max(imageCorners, axis=0)
                    cv2.rectangle(
                        image,
                        (int(min_x), int(min_y)),
                        (int(max_x), int(max_y)),
                        (255, 255, 255),
                        2,
                    )

                    for imageCorner in imageCorners:
                        uC, uV = imageCorner
                        cv2.circle(image, (int(uC), int(uV)), 3, (255, 255, 255), 2)

                    cv2.putText(
                        image,
                        f"{offset_idx}",
                        (int(u), int(v) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                onScreenBranches[offset_idx] = (u, v)

            coordinates[output.getId()] = onScreenBranches

        return coordinates
