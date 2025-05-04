import math
import os
from typing import Union
import cv2
import time
from functools import partial

import numpy as np

# from JXTABLES.XDashDebugger import XDashDebugger

from Core.Agents.Abstract.CSVSimulatedAgentBase import CsvSimulatedAgentBase
from abstract.Capture import Capture, ConfigurableCapture
from abstract.depthCamera import depthCamera
from coreinterface.DetectionPacket import DetectionPacket
from tools.Constants import InferenceMode, CameraExtrinsics, CameraIntrinsics
from mapinternals.localFrameProcessor import LocalFrameProcessor
import Core


class CsvDemoObjectLocalizingAgent(CsvSimulatedAgentBase):
    """Agent -> (CameraUsingAgentBase, PositionLocalizingAgentBase) -> TimestampRegulatedAgentBase -> ObjectLocalizingAgentBase

    Adds inference and object localization capabilites to an agent, processing frames and sending detections
    NOTE: Requires extra arguments passed in somehow, for example using Functools partial or extending the class"""

    DETECTIONPOSTFIX = "Detections"

    def __init__(self, **kwargs) -> None:
        self.cameraIntrinsics = kwargs.get("cameraIntrinsics", None)
        self.cameraExtrinsics = kwargs.get("cameraExtrinsics", None)
        self.inferenceMode = kwargs.get("inferenceMode", None)
        super().__init__(**kwargs)

    def create(self):
        super().create()
        # self.xdashDebugger = XDashDebugger()
        self.Sentinel.info("Creating Frame Processor...")
        currentCoreINFName = self.xclient.getString(Core.COREMODELTABLE)
        currentCoreINFMode = InferenceMode.getFromName(currentCoreINFName, default=None)
        if currentCoreINFMode is not None:
            # assert you are running same model type as any current core process
            isMatch = InferenceMode.assertModelType(
                currentCoreINFMode, self.inferenceMode
            )
            if not isMatch:
                self.Sentinel.fatal(
                    f"Model type mismatch!: Core is Running: {currentCoreINFMode.getModelType()} This is running {self.inferenceMode.getModelType()}"
                )
                raise Exception(
                    f"Model type mismatch!: Core is Running: {currentCoreINFMode.getModelType()} This is running {self.inferenceMode.getModelType()}"
                )
            else:
                self.Sentinel.fatal(f"Model type matched!")
        else:
            self.Sentinel.warning(
                "Was not able to get core model type! Make sure you match!"
            )

        self.frameProcessor = LocalFrameProcessor(
            cameraIntrinsics=self.cameraIntrinsics,
            cameraExtrinsics=self.cameraExtrinsics,
            inferenceMode=self.inferenceMode,
            depthMode=self.depthEnabled,
        )

    def runPeriodic(self) -> None:
        super().runPeriodic()
        with self.timer.run("frame-processing"):
            processedResults = self.frameProcessor.processFrame(
                self.latestFrameMain,
                self.latestFrameDEPTH if self.depthEnabled else None,
                robotPosXCm=self.robotPose2dCMRAD[0],
                robotPosYCm=self.robotPose2dCMRAD[1],
                robotYawRad=self.robotPose2dCMRAD[2],
                drawBoxes=True,
                # if you are sending frames, you likely want to see bounding boxes aswell
            )

    def getDescription(self) -> str:
        return "Inference_Then_Localize_CSVSim"

    def getIntervalMs(self) -> int:
        return 0


def partialCsvDemoObjectLocalizingAgent(
    capture: Capture,
    cameraExtrinsics: CameraExtrinsics,
    cameraIntrinsics: CameraIntrinsics,
    inferenceMode: InferenceMode,
    robotLogCsvPath: str,
    csvAlignmentOffsetS: float,
    isOnRobot: bool = True,
):
    """Returns a partially completed frame processing agent. All you have to do is pass it into neo"""
    return partial(
        CsvDemoObjectLocalizingAgent,
        capture=capture,
        cameraIntrinsics=cameraIntrinsics,
        cameraExtrinsics=cameraExtrinsics,
        inferenceMode=inferenceMode,
        robotLogCsvPath=robotLogCsvPath,
        csvAlignmentOffsetS=csvAlignmentOffsetS,
        isOnRobot=isOnRobot,
        showFrames=True,
    )
