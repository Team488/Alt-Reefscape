import math
import cv2
import numpy as np
from tools.Constants import (
    CameraIntrinsicsPredefined,
    ColorCameraExtrinsics2024,
    MapConstants,
    ATLocations,
)
from tools.Units import LengthType, RotationType
from demos.utils import drawRobotWithCams
from reefTracking.reefPositioner import ReefPositioner
from reefTracking.aprilTagSolver import AprilTagSover
from reefTracking import aprilTagSolver
from tools import UnitConversion, Calculator
from scipy.spatial.transform import Rotation


def visualize_transform(frame, transform, color=(0, 255, 0), length=50) -> None:
    """
    Visualize the position and orientation of a transform matrix on the image,
    drawing from the origin (0, 0) to the transformed position and orientation.

    transform: 4x4 transformation matrix
    frame: The image where the visualization will be drawn
    color: Color of the lines and points to visualize the transform
    length: Length of the direction arrow (for orientation)
    """
    # Get position (translation part of the matrix)
    position = transform[:3, 3]

    # Draw the position as a point (from origin to position)
    cv2.circle(frame, (int(position[0]), int(position[1])), 5, color, -1)

    # Draw orientation as an arrow (use direction from rotation matrix)
    # Take first column of rotation matrix (direction of X axis)
    # orientation = transform[:3, :3] @ np.array([length, 0, 0])  # Apply rotation to the X-axis unit vector
    # end_point = orientation  # End point of the arrow, relative to origin
    cv2.arrowedLine(
        frame,
        (0, 0),  # Origin point (0, 0)
        (int(position[0]), int(position[1])),
        color,
        2,
    )


def getIsBlue(robotPosX, bReefX, rReefX, robotYaw):
    if robotPosX < bReefX:
        return True
    if robotPosX > rReefX:
        return False

    return robotYaw > math.pi


def startDemo() -> None:
    size_x = MapConstants.fieldWidth.getCM()
    size_y = MapConstants.fieldHeight.getCM()
    positioner = ReefPositioner()

    robot_pos = (0, 0)
    robot_width = MapConstants.robotWidth.getCM()
    robot_height = MapConstants.robotHeight.getCM()

    b_reef_center = UnitConversion.toint(MapConstants.b_reef_center.getCM())
    r_reef_center = UnitConversion.toint(MapConstants.r_reef_center.getCM())
    reef_radius = UnitConversion.toint(MapConstants.reefRadius.getCM())

    robotcam_extr = ColorCameraExtrinsics2024.FRONTRIGHT
    robotcam_intr = CameraIntrinsicsPredefined.OV9782COLOR
    solver = AprilTagSover(robotcam_extr, robotcam_intr)

    title = "reef_point_demo"
    rot_trackbar_name = "robot rot deg"
    cv2.namedWindow(title)
    cv2.createTrackbar(rot_trackbar_name, title, 0, 360, lambda x: None)

    def hover_callback(event, x, y, flags, param) -> None:
        nonlocal robot_pos
        robot_pos = (x, y)

    cv2.setMouseCallback(title, hover_callback)
    while True:
        robot_rot = math.radians(cv2.getTrackbarPos(rot_trackbar_name, title))
        preferBlue = getIsBlue(
            robot_pos[0], b_reef_center[0], r_reef_center[0], robot_rot
        )

        frame = np.zeros((size_y, size_x, 3), dtype=np.uint8)

        # draw reefs
        cv2.circle(frame, b_reef_center, reef_radius, (255, 0, 0), 1)
        cv2.circle(frame, r_reef_center, reef_radius, (0, 0, 255), 1)

        drawRobotWithCams(
            frame,
            robot_width,
            robot_height,
            robot_pos[0],
            robot_pos[1],
            robot_rot,
            [(robotcam_extr, robotcam_intr)],
            cameraLineLength=500,
        )

        res1 = positioner.getPostCoordinatesWconst(
            True, robot_pos, robot_rot, robotcam_extr, robotcam_intr
        )
        res2 = positioner.getPostCoordinatesWconst(
            False, robot_pos, robot_rot, robotcam_extr, robotcam_intr
        )
        results = [res1, res2]
        for res in results:
            isBlueSide = res == res1
            focused_reef_center = b_reef_center if isBlueSide else r_reef_center
            if res is None:
                continue
            # draw two "posts"
            x, y, ang, postidxs = res
            coord = 20
            for postidx in postidxs:
                atID = ReefPositioner.getAprilTagId(postidx, isBlueSide)
                pose = ATLocations.get_pose_by_id(atID)

                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(
                    frame, f"P Idx: {postidx}", (10, coord), 1, 1, (255, 255, 255), 1
                )
                coord += 20
                cv2.putText(
                    frame, f"AT Idx: {atID}", (10, coord), 1, 1, (255, 255, 255), 1
                )
                coord += 20

                reefScreenCoordinates = UnitConversion.toint(
                    UnitConversion.convertLength(pose[0], LengthType.IN, LengthType.CM)
                )
                yaw = math.radians(pose[1][0])
                atWidth = 16.51  # cm
                x0, y0 = reefScreenCoordinates[:2]
                y0 = UnitConversion.invertY(y0)
                xd = int(math.sin(yaw) * atWidth)
                yd = int(math.cos(yaw) * atWidth)
                cv2.line(
                    frame, (x0 - xd, y0 - yd), (x0 + xd, y0 + yd), (255, 255, 255), 2
                )

                # draw "posts"
                ang = (
                    math.radians(60) * postidx
                )  # assuming first post starts at angle 0
                offset = math.radians(10)
                Vx1 = focused_reef_center[0] + reef_radius * math.cos(ang - offset)
                Vy1 = focused_reef_center[1] + reef_radius * math.sin(ang - offset)
                cv2.circle(frame, (int(Vx1), int(Vy1)), 5, (255, 192, 203), 1)
                Vx2 = focused_reef_center[0] + reef_radius * math.cos(ang + offset)
                Vy2 = focused_reef_center[1] + reef_radius * math.sin(ang + offset)
                cv2.circle(frame, (int(Vx2), int(Vy2)), 5, (255, 192, 203), 1)

                # x = focused_reef_center[0] + reef_radius * math.cos(fullAng)
                # y = focused_reef_center[1] + reef_radius * math.sin(fullAng)
                # # draw actual angle seen
                # cv2.circle(frame,(int(x),int(y)),5,(0,0,255),1)

                # cam_reef,idx = solver.getNearestAtPose((robot_pos[0],robot_pos[1],robot_rot))
                # if cam_reef is not None:
                #     visualize_transform(frame, cam_reef, color=(0, 0, 255))  # AprilTag in blue
                #     cv2.putText(frame, "AprilTag", (int(cam_reef[0, 3]), int(cam_reef[1, 3])),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # print(cam_reef)

        cv2.imshow(title, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
