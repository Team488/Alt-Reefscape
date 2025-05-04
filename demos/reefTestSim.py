import math
import struct
import time
import queue
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor
from networktables import NetworkTables
import numpy as np
from tools.NtUtils import getPose2dFromBytes
from reefTracking.reefPositioner import ReefPositioner

# from mapinternals.CentralProcessor import CentralProcessor # fix import creating tensorflow
from tools.Constants import (
    ColorCameraExtrinsics2024,
    CameraIntrinsicsPredefined,
    MapConstants,
)
from demos.utils import drawRobotWithCams


def startDemo() -> None:
    processName = "Simulation_Process"
    logger = logging.getLogger(processName)

    # MJPEG stream URLs
    extrinsics = [
        ColorCameraExtrinsics2024.FRONTRIGHT,
        # CameraExtrinsics.FRONTLEFT,
        # CameraExtrinsics.REARRIGHT,
        # CameraExtrinsics.REARLEFT,
    ]
    cams = [(extr, CameraIntrinsicsPredefined.SIMULATIONCOLOR) for extr in extrinsics]

    NetworkTables.initialize(server="192.168.0.17")
    postable = NetworkTables.getTable("SmartDashboard/VisionSystemSim-main/Sim Field")
    table = NetworkTables.getTable("AdvantageKit/RealOutputs/Odometry")

    # Window setup for displaying the camera feed
    title = "Camera_View"
    cv2.namedWindow(title)
    cv2.createTrackbar("t", title, 0, 100, lambda x: None)
    camera_selector_name = "Camera Selection"
    positioner = ReefPositioner()

    try:
        while True:
            pos = (0, 0, 0)  # x(cm), y(cm), rot(rad)
            raw_data = postable.getEntry("Robot").get()
            if raw_data:
                pos = (
                    raw_data[0] * 100,
                    MapConstants.fieldHeight.getCM() - raw_data[1] * 100,
                    -(math.radians(raw_data[2])) + math.pi,
                )  # this one gives degrees by default
                # print(f"{raw_data[2]=} {pos[2]=}")
                # pos = getPose2dFromBytes(raw_data)

                frame = np.zeros(
                    (
                        MapConstants.fieldHeight.getCM(),
                        MapConstants.fieldWidth.getCM(),
                        3,
                    ),
                    dtype=np.uint8,
                )
                drawRobotWithCams(
                    frame,
                    MapConstants.robotWidth.getCM(),
                    MapConstants.robotHeight.getCM(),
                    pos[0],
                    pos[1],
                    pos[2],
                    cams,
                )
                cv2.circle(
                    frame,
                    tuple(map(int, MapConstants.b_reef_center.getCM())),
                    int(MapConstants.reefRadius.getCM()),
                    (0, 255, 0),
                    1,
                )
                post_id = ""
                for cam in cams:
                    res = positioner.getPostCoordinatesWconst(
                        True, pos[:2], pos[2], cam[0], cam[1]
                    )
                    if res:
                        x, y, ang, postIdx = res
                        cv2.circle(
                            frame,
                            (int(x), int(y)),
                            int(MapConstants.reef_post_radius.getCM() + 20),
                            (255, 255, 255),
                            -1,
                        )
                        post_id += f" {postIdx=}"
                cv2.putText(frame, post_id, (10, 20), 1, 1, (255, 255, 255), 1)
                cv2.imshow(title, frame)

            else:
                logger.warning("Cannot get robot location from network tables!")

            # Handle keyboard interrupt with cv2.waitKey()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
