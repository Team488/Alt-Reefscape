import random
import time
import cv2
import numpy as np

from ..Tracking.deepSortBaseLabler import DeepSortBaseLabler
from ..Constants.Inference import InferenceMode, ConfigConstants

from Alt.Cameras.Parameters import CameraIntrinsics, CameraExtrinsics

from .positionEstimator import PositionEstimator
from tools.depthBasedPositionEstimator import DepthBasedPositionEstimator
from .positionTranslations import CameraToRobotTranslator, transformWithYaw
from inference.MultiInferencer import MultiInferencer
from demos import utils
from Core import getChildLogger


Sentinel = getChildLogger("Local_Frame_Processor")


class LocalFrameProcessor:
    """This handles the full pipline from a frame to detections with deepsort id's. You can think of it as the local part of the detection pipeline
    After this detections are centralized over the network to an orin and thats where the Ukf and etc will reside
    """

    def __init__(
        self,
        cameraIntrinsics: CameraIntrinsics,
        cameraExtrinsics: CameraExtrinsics,
        inferenceMode: InferenceMode,
        depthMode=False,
    ) -> None:
        self.depthMode = depthMode
        self.inferenceMode = inferenceMode
        self.inf = self.createInferencer(inferenceMode)
        self.labels = self.inferenceMode.getLabelsAsStr()

        self.baseLabler: DeepSortBaseLabler = DeepSortBaseLabler(
            inferenceMode.getLabelsAsStr()
        )
        self.cameraIntrinsics: CameraIntrinsics = cameraIntrinsics
        self.cameraExtrinsics: CameraExtrinsics = cameraExtrinsics
        if self.depthMode:
            self.estimatorDepth = DepthBasedPositionEstimator()
        else:
            self.estimator = PositionEstimator()
        self.translator = CameraToRobotTranslator()
        self.colors = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(15)
        ]

    def createInferencer(self, inferenceMode: InferenceMode):
        Sentinel.info("Creating inferencer: " + inferenceMode.getName())
        return MultiInferencer(inferenceMode)

    def processFrame(
        self,
        colorFrame,
        depthFrameMM=None,
        robotPosXCm=0,
        robotPosYCm=0,
        robotPosZCm=0,
        robotYawRad=0,
        drawBoxes=False,
        useAbsolutePosition=True,
        customCameraExtrinsics: CameraExtrinsics = None,
        customCameraIntrinsics: CameraIntrinsics = None,
        maxDetections=None,
    ) -> list[list[int, tuple[int, int, int], float, bool, np.ndarray]]:
        if depthFrameMM is None and self.depthMode:
            raise ValueError(
                "When depth mode is enabled, you must provide a depth frame!"
            )

        """output is list of id,(absX,absY,absZ),conf,isRobot,features"""
        camIntrinsics = (
            customCameraIntrinsics
            if customCameraIntrinsics is not None
            else self.cameraIntrinsics
        )
        camExtrinsics = (
            customCameraExtrinsics
            if customCameraExtrinsics is not None
            else self.cameraExtrinsics
        )
        startTime = time.time()
        yoloResults = self.inf.run(
            colorFrame, minConf=ConfigConstants.confThreshold, drawBoxes=False
        )  # we will draw deepsort tracked boxes instead
        if maxDetections != None:
            yoloResults = yoloResults[:maxDetections]

        if len(yoloResults) == 0:
            if drawBoxes:
                endTime = time.time()
                fps = 1 / (endTime - startTime)
                cv2.putText(colorFrame, f"FPS:{fps}", (10, 20), 0, 1, (0, 255, 0), 2)
            return []

        # id(unique),bbox,conf,isrobot,features,
        labledResults = self.baseLabler.labelResults(colorFrame, yoloResults)

        if drawBoxes:
            # draw a box with id,conf and relative estimate
            for labledResult in labledResults:
                id = labledResult[0]
                bbox = labledResult[1]
                conf = labledResult[2]
                classId = labledResult[3]

                label = "INVALID"  # technically redundant, as the deepsort step filters out any invalid class_idxs
                if 0 <= classId < len(self.labels):
                    label = f"{self.labels[classId]} Id:{id}"

                color = self.colors[id % len(self.colors)]
                utils.drawBox(colorFrame, bbox, label, conf, color)

        # id(unique),estimated x/y,conf,class_idx,features,
        if self.depthMode:
            relativeResults = self.estimatorDepth.estimateDetectionPositions(
                colorFrame,
                depthFrameMM,
                labledResults.copy(),
                camIntrinsics,
                self.inferenceMode,
            )
        else:
            relativeResults = self.estimator.estimateDetectionPositions(
                colorFrame, labledResults.copy(), camIntrinsics, self.inferenceMode
            )

        # print(f"{robotPosXCm=} {robotPosYCm=} {robotYawRad=}")
        if useAbsolutePosition:
            finalResults = []
            for result in relativeResults:
                ((relCamX, relCamY)) = result[1]
                (
                    relToRobotX,
                    relToRobotY,
                    relToRobotZ,
                ) = self.translator.turnCameraCoordinatesIntoRobotCoordinates(
                    relCamX, relCamY, camExtrinsics
                )
                # factor in robot orientation
                result[1] = transformWithYaw(
                    np.array([relToRobotX, relToRobotY, relToRobotZ]), robotYawRad
                )
                # update results with absolute position
                result[1] = np.add(
                    result[1], np.array([robotPosXCm, robotPosYCm, robotPosZCm])
                )

                # note at this point these values are expected to be absolute
                absx, absy, absz = result[1]
                if not self.isiregularDetection(absx, absy, absz):
                    finalResults.append(result)
                else:
                    Sentinel.warning("Iregular Detection!:")
                    Sentinel.debug(f"{absx =} {absy =} {absz =}")
                    Sentinel.debug(f"{relToRobotX =} {relToRobotY =} {relToRobotZ =}")

        else:
            for result in relativeResults:
                result[1].append(0)  # add z component
            finalResults = relativeResults

        # output is id,(absX,absY,absZ),conf,class_idx,features

        endTime = time.time()

        fps = 1 / (endTime - startTime)

        if drawBoxes:
            # add final fps
            cv2.putText(colorFrame, f"FPS:{fps}", (10, 20), 0, 1, (0, 255, 0), 2)

        return finalResults

    def isiregularDetection(self, x, y, z):  # cm
        return (
            x < 0
            or x >= MapConstants.fieldWidth.value
            or y < 0
            or y >= MapConstants.fieldHeight.value
            # or z < -maxDelta
            # or z > maxDelta
        )
