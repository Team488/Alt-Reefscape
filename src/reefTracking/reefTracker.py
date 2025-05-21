from typing import Optional

import cv2
import numpy as np

from Alt.Core import getChildLogger
from Alt.Cameras.Parameters.CameraIntrinsics import CameraIntrinsics

from ..Constants.AprilTags import ATLocations
from ..math import affline4x4utils
from ..tools import load
from .aprilTagHelper import AprilTagLocal



### CAD Offset:
### X = Downwards -> Right
### Y = Upwards -> Right
### Z = Upwards

# Measurement is in Inches

# CAD to tip of the rod. (MAX Distance)
cad_to_branch_offset = {
    0: np.array([-6.756, -19.707, 2.608]),  # "L2-L"
    1: np.array([6.754, -19.707, 2.563]),  # "L2-R"
    2: np.array([-6.639, -35.606, 2.628]),  # "L3-L"
    3: np.array([6.637, -35.606, 2.583]),  # "L3-R"
    4: np.array([-6.470, -58.4175, 0.921]),  # "L4-L"
    5: np.array([6.468, -58.4175, 0.876]),  # "L4-R"
}

cad_to_algae_offset = {  # Rectangle Centers
    1: np.array([-0.001, -24.6565, 2.5955]),  # "L2-L3 Center"
    2: np.array([-0.001, -47.01175, 1.752]),  # "L3-L4 Center"
}

# Convert to meters:
for branch, offset in cad_to_branch_offset.items():
    for i in range(len(offset)):
        offset[i] *= 0.0254

for branch, offset in cad_to_algae_offset.items():
    for i in range(len(offset)):
        offset[i] *= 0.0254

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


widthOffset = 1.5  # in                                                                   #               ↖  (z)
heightOffset = 5  # in                                                                 #                (3)
heightClearence = -1
depthOffset = 6  # in                                                             #                  \
reefBoxOffsetsFRONT = [  # in to m                                                       #                   \
    np.array(
        [widthOffset / 2 * 0.0254, -heightClearence * 0.0254, 0]
    ),  # (1)  (x) <(1)------o
    np.array(
        [-widthOffset / 2 * 0.0254, -heightClearence * 0.0254, 0]
    ),  # (2)                |
    np.array(
        [-widthOffset / 2 * 0.0254, heightOffset * 0.0254, 0]
    ),  # (2)                |
    np.array(
        [widthOffset / 2 * 0.0254, heightOffset * 0.0254, 0]
    ),  # (2)                |
    # np.array([-widthOffset / 2 * 0.0254, heightClearence * 0.0254, zoffset* 0.0254]),  # (3)               (2)
]
reefBoxOffsetsLEFT = [  # in to m
    np.array(
        [widthOffset / 2 * 0.0254, -heightClearence * 0.0254, depthOffset * 0.0254]
    ),  # top left
    np.array([widthOffset / 2 * 0.0254, -heightClearence * 0.0254, 0]),  # top right
    np.array([widthOffset / 2 * 0.0254, heightOffset * 0.0254, 0]),  # bottom right
    np.array(
        [widthOffset / 2 * 0.0254, heightOffset * 0.0254, depthOffset * 0.0254]
    ),  # bottom left
]
reefBoxOffsetsRIGHT = [  # in to m
    np.array([-widthOffset / 2 * 0.0254, -heightClearence * 0.0254, 0]),  # top right
    np.array(
        [-widthOffset / 2 * 0.0254, -heightClearence * 0.0254, depthOffset * 0.0254]
    ),  # top left
    np.array(
        [-widthOffset / 2 * 0.0254, heightOffset * 0.0254, depthOffset * 0.0254]
    ),  # bottom left
    np.array([-widthOffset / 2 * 0.0254, heightOffset * 0.0254, 0]),  # bottom right
]
reefBoxOffsetsTOP = [  # in to m
    np.array(
        [widthOffset / 2 * 0.0254, -heightClearence * 0.0254, depthOffset * 0.0254]
    ),
    np.array(
        [-widthOffset / 2 * 0.0254, -heightClearence * 0.0254, depthOffset * 0.0254]
    ),
    np.array([-widthOffset / 2 * 0.0254, -heightClearence * 0.0254, 0]),
    np.array([widthOffset / 2 * 0.0254, -heightClearence * 0.0254, 0]),
]
reefBoxOffsetsBOTTOM = [  # in to m
    np.array([widthOffset / 2 * 0.0254, heightOffset * 0.0254, depthOffset * 0.0254]),
    np.array([-widthOffset / 2 * 0.0254, heightOffset * 0.0254, depthOffset * 0.02540]),
    np.array([-widthOffset / 2 * 0.0254, heightOffset * 0.0254, 0]),
    np.array([widthOffset / 2 * 0.0254, heightOffset * 0.0254, 0]),
]

reefBoxOffsets = [
    reefBoxOffsetsFRONT,
    reefBoxOffsetsLEFT,
    reefBoxOffsetsRIGHT,
    reefBoxOffsetsTOP,
    reefBoxOffsetsBOTTOM,
]


widthOffset = 8  # in                                                                   #               ↖  (z)
heightOffset = 8  # in                                                                 #                (3)
depthOffset = 6  # in                                                             #                  \
algaeBoxOffsetsFRONT = [  # in to m                                                       #                   \
    np.array(
        [widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, 0]
    ),  # (1)  (x) <(1)------o
    np.array(
        [-widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, 0]
    ),  # (2)                |
    np.array(
        [-widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, 0]
    ),  # (2)                |
    np.array(
        [widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, 0]
    ),  # (2)                |
    # np.array([-widthOffset / 2 * 0.0254, heightClearence * 0.0254, zoffset* 0.0254]),  # (3)               (2)
]
algaeBoxOffsetsLEFT = [  # in to m
    np.array(
        [widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, depthOffset * 0.0254]
    ),  # top left
    np.array([widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, 0]),  # top right
    np.array([widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, 0]),  # bottom right
    np.array(
        [widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, depthOffset * 0.0254]
    ),  # bottom left
]
algaeBoxOffsetsRIGHT = [  # in to m
    np.array([-widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, 0]),  # top right
    np.array(
        [-widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, depthOffset * 0.0254]
    ),  # top left
    np.array(
        [-widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, depthOffset * 0.0254]
    ),  # bottom left
    np.array([-widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, 0]),  # bottom right
]
algaeBoxOffsetsTOP = [  # in to m
    np.array(
        [widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, depthOffset * 0.0254]
    ),
    np.array(
        [-widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, depthOffset * 0.0254]
    ),
    np.array([-widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, 0]),
    np.array([widthOffset / 2 * 0.0254, -heightOffset / 2 * 0.0254, 0]),
]
algaeBoxOffsetsBOTTOM = [  # in to m
    np.array(
        [widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, depthOffset * 0.0254]
    ),
    np.array(
        [-widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, depthOffset * 0.02540]
    ),
    np.array([-widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, 0]),
    np.array([widthOffset / 2 * 0.0254, heightOffset / 2 * 0.0254, 0]),
]

algaeBoxOffsets = [
    algaeBoxOffsetsFRONT,
    algaeBoxOffsetsLEFT,
    algaeBoxOffsetsRIGHT,
    algaeBoxOffsetsTOP,
    algaeBoxOffsetsBOTTOM,
]


def getClosest3FacesCoral(tag_to_cam_translation, frame):
    pitch, yaw, _ = affline4x4utils.extract_angles(
        tag_to_cam_translation, format="XYZ"
    )
    # cv2.putText(
    #     frame, f"h: {yaw:.2f} v: {pitch:.2f}", (0, 40), 0, 1, (255, 255, 255), 1
    # )
    horizontal_boxOffset = reefBoxOffsetsLEFT if yaw > 0 else reefBoxOffsetsRIGHT
    vertical_boxOffset = reefBoxOffsetsTOP if pitch < 0 else reefBoxOffsetsBOTTOM
    return [reefBoxOffsetsFRONT, horizontal_boxOffset, vertical_boxOffset]


def getClosest3FacesAlgae(tag_to_cam_translation, frame):
    pitch, yaw, _ = affline4x4utils.extract_angles(
        tag_to_cam_translation, format="XYZ"
    )
    # cv2.putText(
    #     frame, f"h: {yaw:.2f} v: {pitch:.2f}", (0, 40), 0, 1, (255, 255, 255), 1
    # )
    horizontal_boxOffset = algaeBoxOffsetsLEFT if yaw > 0 else algaeBoxOffsetsRIGHT
    vertical_boxOffset = algaeBoxOffsetsTOP if pitch < 0 else algaeBoxOffsetsBOTTOM
    return [algaeBoxOffsetsFRONT, horizontal_boxOffset, vertical_boxOffset]


def transform_basis_from_frc_toimg(T):
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


def backProjWhite(labImage, threshold=120):
    # return cv2.calcBackProject([bumperOnlyLab],[1,2],whiteNumHist,[0,256,0,256],1)
    L, a, b = cv2.split(labImage)

    # Threshold the L channel to get a binary image
    # Here we assume white has high L values, you might need to adjust the threshold value
    _, white_mask = cv2.threshold(L, threshold, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((5, 5), np.uint8)
    # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    return white_mask



purpleHist = load.load_asset_numpy(
    "histograms", "reef_post_hist.npy"
)
whiteHist = load.load_asset_numpy(
    "histograms", "whiteCoralHistBAD.npy"
)
algaeHist = load.load_asset_numpy(
    "histograms", "blueAlgaeHist.npy"
)

objThresh = 0.2
blockerThresh = 0.2
fullpurpleThresh = 0.7

Sentinel = getChildLogger("Reef_Post_Estimator")
AtCorrectionMap = {}


class ReefTracker:
    def __init__(
        self,
        cameraIntrinsics: CameraIntrinsics,
    ) -> None:
        """
        Creates a reef post estimator, that using april tag results will detect coral slots and probabilistically measure if they are occupied\n
        This works by using a color camera along with known extrinsics/intrinsics and known extrinsics for the april tag camera

        """
        self.camIntr = cameraIntrinsics
        self.ATPoseGetter = AprilTagLocal(cameraIntrinsics)

    def __isInFrame(self, u, v):
        return 0 <= u < self.camIntr.getHres() and 0 <= v < self.camIntr.getVres()

    def getAllTracks(self, colorframe, drawBoxes=True):

        allTracksCoral = {}
        allTracksAlgae = {}
        greyFrame = cv2.cvtColor(colorframe, cv2.COLOR_BGR2GRAY)
        atDetections = self.ATPoseGetter.getDetections(greyFrame)
        atPoses = self.ATPoseGetter.getOrthogonalEstimates(atDetections)
        for detection, pose in zip(atDetections, atPoses):
            tracks = self.__getTracksForPost(
                colorframe, pose.pose1.toMatrix(), drawBoxes, detection.getId()
            )
            centerX = detection.getCenter().x
            centerY = detection.getCenter().y
            if drawBoxes:
                cv2.putText(
                    colorframe,
                    f"{pose.pose1.toMatrix()}",
                    tuple(map(int, (centerX, centerY))),
                    1,
                    1,
                    (255, 255, 255),
                    1,
                )

            if tracks is not None:
                coralTracks, algaeOccuppancy = tracks
                allTracksCoral[detection.getId()] = coralTracks
                if algaeOccuppancy is not None:
                    allTracksAlgae[detection.getId()] = algaeOccuppancy

        return allTracksCoral, allTracksAlgae, list(zip(atDetections, atPoses))

    def __getTracksForPost(self, colorframe, tagPoseMatrix, drawBoxes, atId):
        if (
            tagPoseMatrix is None
            or np.isclose(tagPoseMatrix[:3, 3], np.zeros(shape=(3))).all()
        ):
            Sentinel.warning(f"Invalid tag pose matrix!: {tagPoseMatrix}")
            return None

        correctionOffset = np.array(AtCorrectionMap.get(atId, [0, 0, 0]))
        coralOpenBranches = {}
        algaeLevel = ATLocations.getAlgaeLevel(atId)
        if algaeLevel is None:
            Sentinel.warning(f"Invalid april tag for posts!: {atId}")
            return None

        # could be either high or low
        algaeOffset = cad_to_algae_offset[algaeLevel]

        # iterate over each branch of reef
        for offset_idx, offset_3d in cad_to_branch_offset.items():
            openPercentage = self.__runTrack(
                tagPoseMatrix,
                np.add(offset_3d, correctionOffset),
                getClosest3FacesCoral(tagPoseMatrix, colorframe),
                colorframe,
                purpleHist,
                drawBoxes,
                isAlgae=False,
            )
            if openPercentage is not None:
                # dont add of unknown
                coralOpenBranches[offset_idx] = openPercentage

        # run track on the one algae slot each side of the reef has
        algaeOcupancy = self.__runTrack(
            tagPoseMatrix,
            np.add(algaeOffset, correctionOffset),
            getClosest3FacesAlgae(tagPoseMatrix, colorframe),
            colorframe,
            algaeHist,
            drawBoxes,
            isAlgae=True,
        )

        return coralOpenBranches, algaeOcupancy

    def __runTrack(
        self,
        tagPoseMatrix,
        offset_3d,
        reefBoxOffsets,
        colorframe,
        objectHist,
        drawBoxes,
        isAlgae=False,
    ) -> Optional[float]:
        # solve camera -> branch via camera -> tag and tag -> branch transformations
        frameCopy = colorframe.copy()

        tag_to_reef_homography = np.append(offset_3d, 1.0)  # ensures shape is 4x4

        color_camera_to_reef = np.dot(
            tagPoseMatrix,
            tag_to_reef_homography,
        )

        total_color_corners = []
        for reefBoxOffset in reefBoxOffsets:
            color_corners = []
            for cornerOffset in reefBoxOffset:
                box_offset_homogeneous = np.append(cornerOffset, 0)  # Shape: (4,)

                tag_to_reef_corner_homography = (
                    tag_to_reef_homography + box_offset_homogeneous
                )

                color_camera_to_reef_corner = np.dot(
                    tagPoseMatrix,
                    tag_to_reef_corner_homography,
                )
                color_corners.append(color_camera_to_reef_corner)

            total_color_corners.append(color_corners)

        # project the 3D reef point to 2D image coordinates:
        x_cam, y_cam, z_cam, _ = color_camera_to_reef

        u = (self.camIntr.getFx() * x_cam / z_cam) + self.camIntr.getCx()
        v = (self.camIntr.getFy() * y_cam / z_cam) + self.camIntr.getCy()

        if not self.__isInFrame(u, v):
            return None
        # print(f"{u=} {v=}")

        # project the 3d box corners to 2d image coords
        reef_mask = np.zeros_like(frameCopy)
        total_image_corners = []
        for color_corner in total_color_corners:
            imageCorners = []
            for corner in color_corner:
                x_cam, y_cam, z_cam, _ = corner
                uC = (self.camIntr.getFx() * x_cam / z_cam) + self.camIntr.getCx()
                uV = (self.camIntr.getFy() * y_cam / z_cam) + self.camIntr.getCy()
                imageCorners.append((int(uC), int(uV)))

            total_image_corners.append(imageCorners)
            # Fill the polygon (the rectangle area) with white (255)
            cv2.fillPoly(reef_mask, [np.int32(imageCorners)], (255, 255, 255))

        # base sum
        extracted = cv2.bitwise_and(frameCopy, reef_mask)
        grayExtract = cv2.cvtColor(
            extracted, cv2.COLOR_BGR2GRAY
        )  # Shape becomes (w, h)
        _, baseThreshAll = cv2.threshold(grayExtract, 1, 255, cv2.THRESH_BINARY)
        totalSum = np.sum(baseThreshAll >= 1)

        # color sum
        lab = cv2.cvtColor(extracted, cv2.COLOR_BGR2LAB)
        backProj = cv2.calcBackProject([lab], [1, 2], objectHist, [0, 256, 0, 256], 1)
        dil = cv2.dilate(backProj, np.ones((2, 2)), iterations=5)
        _, threshObject = cv2.threshold(dil, 50, 255, cv2.THRESH_BINARY)
        sumColor = np.sum(threshObject >= 1)
        percentColor = sumColor / totalSum

        # optional blocker sum # eg check if its actually coral blocking (not used for algae)
        if not isAlgae:
            backProjBlocker = backProjWhite(lab, threshold=80)
            _, threshBlocker = cv2.threshold(
                backProjBlocker, 20, 255, cv2.THRESH_BINARY
            )
            sumBlocker = np.sum(threshBlocker >= 1)
            percBlocker = sumBlocker / totalSum

        colorPercentage = 0  # default is 0 eg zero confidence of color
        color = (0, 0, 255)  # red eg not found

        if percentColor > objThresh:
            color = (0, 255, 0)  # green eg color met
            colorPercentage = percentColor / fullpurpleThresh

        elif not isAlgae and percBlocker < blockerThresh:
            colorPercentage = None  #
            color = (0, 0, 0)  # black eg blocked by unknown object

        if drawBoxes:
            cv2.circle(colorframe, (int(u), int(v)), 3, color, 2)
            for imageCorner in total_image_corners:
                for point1 in imageCorner:
                    for point2 in imageCorner:
                        if point1 != point2:
                            cv2.line(colorframe, point1, point2, color, 1)

                for imageCorner in imageCorner:
                    uC, uV = imageCorner
                    cv2.circle(colorframe, (int(uC), int(uV)), 1, color, 2)

            if colorPercentage is not None:
                colorPercentage = np.clip(colorPercentage, -1, 1)

        return colorPercentage
