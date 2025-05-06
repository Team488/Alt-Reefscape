import math
import cv2
import numpy as np
from Core.Agents.Abstract import CameraUsingAgentBase
from Captures.FileCapture import FileCapture
from functools import partial

from abstract.AlignmentProvider import AlignmentProvider


class AlignmentProviderAgent(CameraUsingAgentBase):
    DEFAULTTHRESH = 10

    def __init__(
        self,
        alignmentProvider: AlignmentProvider,
        cameraPath="http://localhost:1181/stream.mjpg",
        showFrames=False,
        flushCamMs=-1,
    ):
        super().__init__(
            capture=FileCapture(videoFilePath=cameraPath, flushTimeMS=flushCamMs),
            showFrames=showFrames,
        )
        self.alignmentProvider = alignmentProvider

    def create(self):
        super().create()
        self.alignmentProvider._inject(self.propertyOperator)
        self.alignmentProvider.create()

        self.leftDistanceProp = self.propertyOperator.createCustomReadOnlyProperty(
            propertyTable="verticalEdgeLeftDistancePx",
            propertyValue=-1,
            addBasePrefix=False,
            addOperatorPrefix=False,
        )
        self.rightDistanceProp = self.propertyOperator.createCustomReadOnlyProperty(
            propertyTable="verticalEdgeRightDistancePx",
            propertyValue=-1,
            addBasePrefix=False,
            addOperatorPrefix=False,
        )

        self.hresProp = self.propertyOperator.createCustomReadOnlyProperty(
            propertyTable="alignmentCameraHres",
            propertyValue=640,
            addBasePrefix=False,
            addOperatorPrefix=False,
        )
        self.vresProp = self.propertyOperator.createCustomReadOnlyProperty(
            propertyTable="alignmentCameraVres",
            propertyValue=480,
            addBasePrefix=False,
            addOperatorPrefix=False,
        )

    def runPeriodic(self) -> None:
        super().runPeriodic()
        frame = self.latestFrameMain
        left, right = self.alignmentProvider.align(
            frame, self.showFrames or self.sendFrame or self.stream_queue is not None
        )
        self.leftDistanceProp.set(left)
        self.rightDistanceProp.set(right)
        self.hresProp.set(frame.shape[1])
        self.vresProp.set(frame.shape[0])

    def getDescription(self) -> str:
        return "Looks-Through-Camera-Checks-Alignment"


def partialAlignmentCheck(
    alignmentProvider: AlignmentProvider,
    cameraPath="http://localhost:1181/stream.mjpg",
    showFrames=False,
    flushCamMs=-1,
):
    return partial(
        AlignmentProviderAgent,
        alignmentProvider=alignmentProvider,
        cameraPath=cameraPath,
        showFrames=showFrames,
        flushCamMs=flushCamMs,
    )
