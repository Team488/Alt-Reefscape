import math
import cv2
import numpy as np
from tools.Constants import CameraIntrinsics, InferenceMode, Label
from Alt.Core import staticLoad, getChildLogger

Sentinel = getChildLogger("Position_Estimator")


class DepthBasedPositionEstimator:
    def __init__(self) -> None:
        self.__blueRobotHist, _ = staticLoad("histograms/blueRobotHist.npy")
        self.__redRobotHist, _ = staticLoad("histograms/redRobotHist.npy")
        self.__minPerc = 0.2

    def __crop_image(
        self, image: np.ndarray, bbox, safety_margin: float = 0
    ) -> np.ndarray:  # in decimal percentage. Eg 5% margin -> 0.05
        x1, y1, x2, y2 = bbox

        if safety_margin != 0:
            xMax, yMax = image.shape[1], image.shape[0]
            width = x2 - x1
            height = y2 - y1
            x1 = int(np.clip(x1 - safety_margin * width, 0, xMax))
            x2 = int(np.clip(x2 + safety_margin * width, 0, xMax))
            y1 = int(np.clip(y1 - safety_margin * height, 0, yMax))
            y2 = int(np.clip(y2 + safety_margin * height, 0, yMax))

        cropped_image = image[y1:y2, x1:x2]

        return cropped_image

    # returns the backprojected frame with either true for blue for false for red
    # if there was no color at all the frame returned will have a corresponding None value
    def __backprojAndThreshFrame(self, frame, histogram, isBlue):
        backProj = cv2.calcBackProject([frame], [1, 2], histogram, [0, 256, 0, 256], 1)
        # cv2.imshow(f"backprojb b?:{isBlue}",backProj)
        _, thresh = cv2.threshold(backProj, 50, 255, cv2.THRESH_BINARY)
        # cv2.imshow(f"thresh b?:{isBlue}",thresh)

        return thresh

    def __getMajorityWhite(self, thresholded_image, bbox):
        # Calculate the percentage of match pixels
        bumperExtracted = self.__crop_image(thresholded_image, bbox)
        num_match = np.count_nonzero(bumperExtracted)

        matchPercentage = num_match / bumperExtracted.size
        return matchPercentage

    """ Checks a frame for two backprojections. Either a blue or red bumper. If there is enough of either color, then its a sucess and we return the backprojected value. Else a fail"""

    def __backprojCheck(self, frame, redHist, blueHist, bbox):
        redBackproj = self.__backprojAndThreshFrame(frame, redHist, False)
        blueBackproj = self.__backprojAndThreshFrame(frame, blueHist, True)
        # cv2.imshow("Blue backproj",blueBackproj)
        redPerc = self.__getMajorityWhite(redBackproj, bbox)
        bluePerc = self.__getMajorityWhite(blueBackproj, bbox)

        if redPerc > bluePerc:
            if redPerc > self.__minPerc:
                Sentinel.debug("Red suceess")
                return (redBackproj, False)
            else:
                # failed minimum percentage
                Sentinel.debug(f"Red fail perc : {redPerc}")
                return (None, None)
        else:
            # blue greater
            if bluePerc > self.__minPerc:
                Sentinel.debug("blue sucess")
                return (blueBackproj, True)
            else:
                Sentinel.debug(f"Blue fail perc : {bluePerc}")
                return (None, None)

    def __getCentralDepthEstimateCM(self, depthFrameMM: np.ndarray, bbox, batch=5):
        # todo find calibrated values for other cams
        centerPoint = np.divide(np.add(bbox[:2], bbox[2:]), 2)
        x, y = map(int, centerPoint)
        mx = max(0, x - batch)
        my = max(0, y - batch)
        lx = min(depthFrameMM.shape[1], y + batch)
        ly = min(depthFrameMM.shape[0], y + batch)
        return np.mean(depthFrameMM[my:ly][mx:lx]) / 10

    """ calculates angle change per pixel, and multiplies by number of pixels off you are. Dimensions either rad or deg depending on ivput fov

    """

    def __calcBearing(self, fov, res, pixelDiff):
        fovperPixel = fov / res
        return -pixelDiff * fovperPixel

    def __getRobotDepthCMCOLOR(
        self, depthFrameMM: np.ndarray, colorFrame: np.ndarray, bbox
    ) -> tuple[float, bool]:
        """Isolates robot bumper based on color, and then gets the horizontal center of the bumper"""
        # Convert to LAB color space
        labFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2LAB)
        processed, isBlue = self.__backprojCheck(
            labFrame, self.__redRobotHist, self.__blueRobotHist, bbox
        )
        if isBlue is None:
            return None
        # Adjust kernel size and iterations based on frame size
        bumperKernel = np.ones((2, 2), np.uint8)
        iterations_close = 1
        iterations_open = 1

        # Morphological operations for bumper
        bumper_closed = cv2.morphologyEx(
            processed, cv2.MORPH_CLOSE, bumperKernel, iterations=iterations_close
        )
        bumper_opened = cv2.morphologyEx(
            bumper_closed, cv2.MORPH_OPEN, bumperKernel, iterations=iterations_open
        )

        _, thresh = cv2.threshold(bumper_opened, 50, 255, cv2.THRESH_BINARY)
        thresh_mask = thresh == 255

        depth_masked = depthFrameMM[thresh_mask]
        # mm->cm
        average_depth = np.mean(depth_masked) / 10 if depth_masked.size > 0 else None
        return average_depth

    def __getRobotDepthCM(self, depthFrameMM: np.ndarray, bbox) -> tuple[float, bool]:
        """Isolates robot bumper based on color, and then gets the horizontal center of the bumper"""
        x1, y1, x2, y2 = bbox
        midX = min(int((x1 + x2) / 2), depthFrameMM.shape[1] - 1)
        botY = y2 - 1
        step = -1

        deltas = np.diff(depthFrameMM[:, midX])

        diffThresh = 10  # looking for less than 10 mm change in direction

        selectedDepth = None
        while botY >= 0:
            delta = deltas[botY]
            if abs(delta) < diffThresh:
                selectedDepth = depthFrameMM[botY, midX]
                cv2.circle(
                    depthFrameMM,
                    center=(midX, botY),
                    radius=2,
                    color=99999,
                    thickness=-1,
                )

                # add depth line here
                dirs = (1, -1)
                deltasHorizontal = np.diff(depthFrameMM[botY, :])
                for dir in dirs:
                    nx = midX + dir
                    while (
                        0 <= nx < len(deltasHorizontal)
                        and abs(deltasHorizontal[nx]) < diffThresh
                    ):
                        cv2.circle(
                            depthFrameMM,
                            center=(nx, botY),
                            radius=2,
                            color=99999,
                            thickness=-1,
                        )

                        depthProbe = depthFrameMM[botY, nx]

                        selectedDepth = min(selectedDepth, depthProbe)
                        nx += dir

                break

            botY += step

        return selectedDepth / 10 if selectedDepth is not None else selectedDepth

    def __estimateRelativeRobotPosition(
        self,
        colorFrame: np.ndarray,
        depthFrameMM: np.ndarray,
        boundingBox,
        cameraIntrinsics: CameraIntrinsics,
    ) -> tuple[float, float]:
        x1, _, x2, _ = boundingBox
        centerX = (x2 + x1) / 2
        depthCM = self.__getRobotDepthCM(depthFrameMM, boundingBox)
        import math

        if depthCM is not None and depthCM and not math.isnan(depthCM):
            bearing = self.__calcBearing(
                CameraIntrinsics.getVfov(cameraIntrinsics, radians=True),
                cameraIntrinsics.getHres(),
                int(centerX - cameraIntrinsics.getCx()),
            )
            Sentinel.debug(f"{depthCM=} {bearing=}")
            estCoords = self.componentizeMagnitudeAndBearing(depthCM, bearing)

            return estCoords

        return None

    def __simpleEstimatePosition(
        self, depthFrameMM: np.ndarray, boundingBox, cameraIntrinsics: CameraIntrinsics
    ) -> tuple[float, float]:
        x1, _, x2, _ = boundingBox
        centerX = (x2 + x1) / 2
        depthCM = self.__getCentralDepthEstimateCM(
            depthFrameMM,
            boundingBox,
        )
        import math

        if depthCM is not None and depthCM and not math.isnan(depthCM):
            bearing = self.__calcBearing(
                CameraIntrinsics.getVfov(cameraIntrinsics, radians=True),
                cameraIntrinsics.getHres(),
                int(centerX - cameraIntrinsics.getCx()),
            )
            Sentinel.debug(f"{depthCM=} {bearing=}")
            estCoords = self.componentizeMagnitudeAndBearing(depthCM, bearing)
            return estCoords
        return None

    def __estimateRelativePosition(
        self,
        class_idx: int,
        colorFrame: np.ndarray,
        depthframeMM: np.ndarray,
        bbox: list,
        cameraIntrinsics: CameraIntrinsics,
        inferenceMode: InferenceMode,
    ):
        labels = inferenceMode.getLabels()
        if class_idx < 0 or class_idx >= len(labels):
            Sentinel.warning(
                f"Estimate relative position got an out of bounds class_idx: {class_idx}"
            )
            return None

        label = labels[class_idx]
        if label == Label.ROBOT:
            return self.__estimateRelativeRobotPosition(
                colorFrame, depthframeMM, bbox, cameraIntrinsics
            )
        if label in {Label.NOTE, Label.ALGAE, Label.CORAL}:
            return self.__simpleEstimatePosition(depthframeMM, bbox, cameraIntrinsics)
        Sentinel.warning(
            f"Label: {str(label)} is not supported for position estimation!"
        )
        return None

    def estimateDetectionPositions(
        self,
        colorFrame: np.ndarray,
        depthframeMM: np.ndarray,
        labledResults,
        cameraIntrinsics: CameraIntrinsics,
        inferenceMode: InferenceMode,
    ):
        if colorFrame.shape[:2] != depthframeMM.shape[:2]:
            Sentinel.fatal(
                f"colorFrame and depth frame shape must match! {colorFrame.shape=} {depthframeMM.shape=}"
            )
            raise ValueError(
                f"colorFrame and depth frame shape must match! {colorFrame.shape=} {depthframeMM.shape=}"
            )

        estimatesOut = [
            [result[0], estimate, result[2], result[3], result[4]]
            for result in labledResults
            if (
                estimate := self.__estimateRelativePosition(
                    result[3],
                    colorFrame,
                    depthframeMM,
                    result[1],
                    cameraIntrinsics,
                    inferenceMode,
                )
            )
        ]

        return estimatesOut

    """ This follows the idea that the distance we calculate is independent to bearing. This means that the distance value we get is the X dist. Thus  y will be calculated using bearing
        Takes hDist, bearing (radians) and returns x,y
    """

    def componentizeHDistAndBearing(self, hDist, bearing):
        x = hDist
        y = math.tan(bearing) * hDist
        return x, y

    def componentizeMagnitudeAndBearing(self, magnitude, bearing):
        x = math.cos(bearing) * magnitude
        y = math.sin(bearing) * magnitude
        return x, y
