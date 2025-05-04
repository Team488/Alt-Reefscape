import numpy as np
from mapinternals.probmap import ProbMap
from mapinternals.localFrameProcessor import LocalFrameProcessor
from tools.CsvParser import CsvParser
from tools.Constants import (
    CameraIntrinsicsPredefined,
    ColorCameraExtrinsics2024,
    InferenceMode,
)
from inference.onnxInferencer import onnxInferencer
import cv2
import math

from tools.Units import UnitMode


def startDemo() -> None:
    cv2.namedWindow("view", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("view", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    fieldWidth = 1653  # 54' 3" to cm
    fieldHeight = 800  # 26' 3" to cm
    res = 5  # cm
    robotWidth = 75  # cm
    robotHeight = 75  # cm assuming square robot with max frame perimiter of 300
    gameObjectWidth = 35  # cm
    gameObjectHeight = 35  # cm circular note
    simMap = ProbMap(
        fieldWidth,
        fieldHeight,
        res,
        gameObjectWidth=gameObjectWidth,
        gameObjectHeight=gameObjectHeight,
        robotWidth=robotWidth,
        robotHeight=robotHeight,
    )

    cameraExtr = ColorCameraExtrinsics2024.NONE
    cameraIntr = CameraIntrinsicsPredefined.OV9782COLOR
    cap = cv2.VideoCapture(0)

    frameProcessor = LocalFrameProcessor(
        cameraIntr, cameraExtr, inferenceMode=InferenceMode.ONNX2024
    )
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # read and draw robot location

            # Run yolov5 on the frame
            out = frameProcessor.processFrame(
                frame, None, simMap.width / 2, simMap.height / 2, 0
            )

            for result in out:
                id = result[0]
                x, y, z = result[1]
                conf = result[2]
                isRobot = result[3]
                cv2.putText(
                    frame,
                    f"X{x-simMap.width/2} Y{y-simMap.height/2}",
                    (20, 50),
                    0,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                if isRobot:
                    simMap.addCustomRobotDetection(int(x), int(y), 200, 200, conf)
                else:
                    simMap.addCustomObjectDetection(int(x), int(y), 200, 200, conf)

            (gameObjMap, robotMap) = simMap.getHeatMaps()
            height, width = robotMap.shape
            zeros = np.zeros((height, width), dtype=np.uint8)
            mapView = cv2.merge((zeros, gameObjMap, robotMap))
            frame = __embed_frame(frame, mapView, scale_factor=1 / 2.7)
            cv2.imshow("view", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                return
            simMap.disspateOverTime(0.2)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def __embed_frame(exterior_frame, interior_frame, scale_factor=1 / 3):
    # Get dimensions of the exterior frame
    exterior_height, exterior_width, _ = exterior_frame.shape

    # Resize the interior frame to be a third of the exterior frame's height
    new_height = int(exterior_height * scale_factor)
    new_width = int(interior_frame.shape[1] * (new_height / interior_frame.shape[0]))
    resized_interior_frame = cv2.resize(interior_frame, (new_width, new_height))

    # Define the position where the smaller frame will be placed (bottom-right corner)
    y_offset = exterior_height - resized_interior_frame.shape[0]
    x_offset = exterior_width - resized_interior_frame.shape[1]

    # Insert the resized interior frame into the exterior frame
    exterior_frame[
        y_offset : y_offset + resized_interior_frame.shape[0],
        x_offset : x_offset + resized_interior_frame.shape[1],
    ] = resized_interior_frame

    return exterior_frame


if __name__ == "__main__":
    print("Must be run from src directory")
    startDemo()
