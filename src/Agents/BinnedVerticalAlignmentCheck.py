from collections import defaultdict
import cv2
import numpy as np
from Core.Agents.Abstract import CameraUsingAgentBase
from Captures.FileCapture import FileCapture
from tools.Constants import SimulationEndpoints
from functools import partial


class BinnedVerticalAlignmentChecker(CameraUsingAgentBase):
    # testHostname = "photonvisionfrontright"  # for testing ONLY
    testHostname = None
    TUNEDWIDTH = 960
    TUNEDHEIGHT = 720

    def __init__(
        self,
        showFrames: bool,
        flushTimeMS: int = -1,
        mjpeg_url: str = "http://localhost:1181/stream.mjpg",
    ):
        super().__init__(
            capture=FileCapture(videoFilePath=mjpeg_url, flushTimeMS=flushTimeMS),
            showFrames=showFrames,
        )

        self.shape = (self.TUNEDWIDTH, self.TUNEDHEIGHT)  # default

    def rescaleWidth(self, value):
        return value * self.shape[1] / self.TUNEDWIDTH

    def rescaleHeight(self, value):
        return value * self.shape[0] / self.TUNEDHEIGHT

    def create(self) -> None:
        super().create()
        if self.testHostname is None:
            self.leftDistanceProp = self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable="verticalEdgeLeftDistancePx",
                propertyValue=-1,
                addBasePrefix=True,
                addOperatorPrefix=False,
            )
            self.rightDistanceProp = self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable="verticalEdgeRightDistancePx",
                propertyValue=-1,
                addBasePrefix=True,
                addOperatorPrefix=False,
            )
            self.hresProp = self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable="cameraHres",
                propertyValue=-1,
                addBasePrefix=True,
                addOperatorPrefix=False,
            )
            self.vresProp = self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable="cameraVres",
                propertyValue=-1,
                addBasePrefix=True,
                addOperatorPrefix=False,
            )
        else:
            self.leftDistanceProp = self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable=f"{self.testHostname}.verticalEdgeLeftDistancePx",
                propertyValue=-1,
                addBasePrefix=False,
                addOperatorPrefix=False,
            )
            self.rightDistanceProp = self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable=f"{self.testHostname}.verticalEdgeRightDistancePx",
                propertyValue=-1,
                addBasePrefix=False,
                addOperatorPrefix=False,
            )
            self.hresProp = self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable=f"{self.testHostname}.cameraHres",
                propertyValue=-1,
                addBasePrefix=False,
                addOperatorPrefix=False,
            )
            self.vresProp = self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable=f"{self.testHostname}.cameraVres",
                propertyValue=-1,
                addBasePrefix=False,
                addOperatorPrefix=False,
            )

        self.sobel_threshold = self.propertyOperator.createProperty(
            propertyTable="inital_sobel_thresh",
            propertyDefault=80,
            setDefaultOnNetwork=True,
        )
        self.threshold_to_last_used = self.propertyOperator.createProperty(
            propertyTable="threshold_to_last_used_size",
            propertyDefault=25,
            setDefaultOnNetwork=True,
        )
        self.bin_size_pixels = self.propertyOperator.createProperty(
            propertyTable="binning_size_pixels",
            propertyDefault=30,
            setDefaultOnNetwork=True,
        )
        self.min_edge_height = self.propertyOperator.createProperty(
            propertyTable="min_vertical_edge_height",
            propertyDefault=250,  # Minimum height in pixels for a valid edge
            setDefaultOnNetwork=True,
        )
        self.distanceMemoryFrames = self.propertyOperator.createProperty(
            propertyTable="number_of_frames_to_keep_memory",
            propertyDefault=30,
            setDefaultOnNetwork=True,
        )
        self.lastUsedSize = None
        self.lastValidLeft = None
        self.lastValidLeftFrameCnt = None
        self.lastValidRight = None
        self.lastValidRightFrameCnt = None
        self.currentFrameCnt = 0

    def runPeriodic(self) -> None:
        super().runPeriodic()
        self.currentFrameCnt += 1

        frame = self.latestFrameMain
        self.shape = frame.shape

        self.vresProp.set(frame.shape[0])
        self.hresProp.set(frame.shape[1])

        newLeftDistance = None
        newRightDistance = None

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.sobel_threshold.get() > 0:
            # this threshold can be though of as a way to only get the april tag lines by first dropping anything other than a dark april tag
            _, blurred = cv2.threshold(
                blurred, self.sobel_threshold.get(), 255, cv2.THRESH_BINARY
            )

        # Use Sobel operator to detect vertical edges (x-direction)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        sobel_8u = np.uint8(abs_sobelx / abs_sobelx.max() * 255)

        # Threshold the edge image
        _, thresh = cv2.threshold(sobel_8u, 100, 255, cv2.THRESH_BINARY)

        if self.showFrames:
            cv2.imshow("thresh", thresh)

        # Apply morphological operations to enhance vertical edges
        kernel_vertical = np.ones((5, 1), np.uint8)
        vertical_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_vertical)

        # Find contours in the vertical edge image
        contours, _ = cv2.findContours(
            vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Prepare a visualization image
        edge_viz = np.zeros_like(frame)

        min_height = self.rescaleHeight(self.min_edge_height.get())
        binSize = self.rescaleHeight(self.bin_size_pixels.get())

        valid_binned = defaultdict(list)

        # match similar heights by binning by a certain resolution, if greater than a threshold height

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter for vertical edges (height > width and minimum height)
            if h > w and h >= min_height:
                bin_idx = h // binSize
                valid_binned[bin_idx].append(contour)

                cv2.rectangle(edge_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # fnd the smallest "pair" of sides that is SILL BIG ENOUGH, but not the biggest in general.
        # This helps distinguish the april tag sides from the side of the april tag paper and the gray background
        bestmatchedPair = None
        sizeLocked = False
        bestsize = -1
        bestPairLength = -1
        for valid_key, valid_bin in valid_binned.items():
            size = valid_key * binSize
            pairLength = min(len(valid_bin), 2)
            if pairLength > bestPairLength:  # prioritize pair then size
                bestmatchedPair = valid_bin[:2]  # ugly, but get only two
                bestPairLength = pairLength
                bestsize = size
            elif pairLength == bestPairLength:
                if self.lastUsedSize is not None:
                    diff = abs(self.lastUsedSize - size)

                    if diff < self.rescaleHeight(self.threshold_to_last_used.get()):
                        bestsize = size
                        bestmatchedPair = valid_bin[:2]
                        sizeLocked = True

                if not sizeLocked and size < bestsize:
                    bestsize = size
                    bestmatchedPair = valid_bin[:2]

        # memory for last used bin size
        self.lastUsedSize = bestsize

        # assign left/right sides
        if bestmatchedPair is not None:
            for biggest in bestmatchedPair:
                x, y, w, h = cv2.boundingRect(biggest)

                # draw valid vertical edge in the visualization
                cv2.drawContours(edge_viz, [biggest], -1, (0, 0, 255), 2)

                mid = x + w / 2

                isLeftEdge = mid < frame.shape[1] / 2

                distMid = int(abs(mid - frame.shape[1] / 2))

                if isLeftEdge:
                    newLeftDistance = distMid
                else:
                    newRightDistance = distMid

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # figure out how Update properties

        # default -1
        selectedLeft = -1
        if newLeftDistance is not None:
            # set memory
            self.lastValidLeft = newLeftDistance
            self.lastValidLeftFrameCnt = self.currentFrameCnt
            selectedLeft = newLeftDistance

        elif self.lastValidLeftFrameCnt is not None:
            # try get memory
            deltaSinceLastValidLeft = self.currentFrameCnt - self.lastValidLeftFrameCnt

            if deltaSinceLastValidLeft < self.distanceMemoryFrames.get():
                # recent enough to put
                selectedLeft = self.lastValidLeft

        # default -1
        selectedRight = -1
        if newRightDistance is not None:
            # set memory
            self.lastValidRight = newRightDistance
            self.lastValidRightFrameCnt = self.currentFrameCnt
            selectedRight = newRightDistance
        elif self.lastValidRightFrameCnt is not None:
            # try get memory
            deltaSinceLastValidRight = (
                self.currentFrameCnt - self.lastValidRightFrameCnt
            )

            if deltaSinceLastValidRight < self.distanceMemoryFrames.get():
                # recent enough to put
                selectedRight = self.lastValidRight

        cv2.putText(
            frame,
            f"L: {selectedLeft}px, R: {selectedRight}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        self.leftDistanceProp.set(selectedLeft)
        self.rightDistanceProp.set(selectedRight)

        # If showing frames is enabled, display the edge visualization
        if self.showFrames:
            cv2.imshow("Vertical Edges", edge_viz)

    def getDescription(self) -> str:
        return "Detects-Vertical-Edges-For-AprilTag-Alignment"


def partialVerticalAlignmentCheck(
    showFrames: bool = False,
    flushTimeMS: int = -1,
    mjpeg_url="http://localhost:1181/stream.mjpg",
):
    return partial(
        BinnedVerticalAlignmentChecker,
        showFrames=showFrames,
        flushTimeMS=flushTimeMS,
        mjpeg_url=mjpeg_url,
    )
