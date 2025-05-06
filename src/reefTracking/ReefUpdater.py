import numpy as np
import cv2
from logging import Logger
from mapinternals.UKF import Ukf
from tools.Constants import InferenceMode, MapConstants, CameraIdOffsets2024
from mapinternals.probmap import ProbMap
from mapinternals.KalmanLabeler import KalmanLabeler
from mapinternals.KalmanCache import KalmanCache
from reefTracking.ReefState import ReefState
from Core import getChildLogger, COREINFERENCEMODE

Sentinel = getChildLogger("Central")


class Central:
    def __init__(
        self,
        inferenceMode: InferenceMode = COREINFERENCEMODE,
    ) -> None:
        self.inferenceMode = inferenceMode
        self.labels = self.inferenceMode.getLabels()

        self.kalmanCaches = [KalmanCache() for _ in self.labels]
        self.objectmap = ProbMap(self.labels)
        self.reefState = ReefState()
        self.ukf = Ukf()
        self.labler = KalmanLabeler(self.kalmanCaches, self.labels)

    def processReefUpdate(
        self,
        reefResults: tuple[list[tuple[int, int, float]], list[tuple[int, float]]],
        timeStepMs,
    ) -> None:
        self.reefState.dissipateOverTime(timeStepMs)

        for reefResult in reefResults:
            # print(reefResult)
            coralObservation, algaeObservation = reefResult
            for apriltagid, branchid, opennessconfidence in coralObservation:
                self.reefState.addObservationCoral(
                    apriltagid, branchid, opennessconfidence
                )

            for apriltagid, opennessconfidence in algaeObservation:
                self.reefState.addObservationAlgae(apriltagid, opennessconfidence)

    def processFrameUpdate(
        self,
        cameraResults: list[
            tuple[
                list[
                    list[int, tuple[int, int, int], float, bool, np.ndarray],
                    int,
                ]
            ]
        ],
        timeStepMs,
    ) -> None:
        # dissipate at start of iteration
        self.objectmap.disspateOverTime(timeStepMs)

        # first get real ids

        # go through each detection and do the magic
        for singleCamResult, idOffset in cameraResults:
            if singleCamResult:
                self.labler.updateRealIds(singleCamResult, idOffset, timeStepMs)
                (id, coord, prob, class_idx, features) = singleCamResult[0]
                # todo add feature deduping here
                (x, y, z) = coord

                # first load in to ukf, (if completely new ukf will load in as new state)
                # index will be filtered out by labler
                self.kalmanCaches[class_idx].LoadInKalmanData(id, x, y, self.ukf)

                newState = self.ukf.predict_and_update([x, y])

                # now we have filtered data, so lets store it. First thing we do is cache the new ukf data
                self.kalmanCaches[class_idx].saveKalmanData(id, self.ukf)

                # input new estimated state into the map

                self.objectmap.addDetectedObject(
                    class_idx, int(newState[0]), int(newState[1]), prob
                )
