from collections import defaultdict
import cv2
import numpy as np
from .AlignmentProvider import AlignmentProvider
from Alt.Core import getChildLogger

Sentinel = getChildLogger("April_Tag_Alignment_Provider")


class BinnedVerticalAlignmentChecker(AlignmentProvider):
    TUNEDWIDTH = 960
    TUNEDHEIGHT = 720

    def __init__(self):
        super().__init__()
        self.shape = (self.TUNEDWIDTH, self.TUNEDHEIGHT)  # default

    def create(self):
        self.createConstants()

    def createConstants(self):
        self.sobel_threshold = self.agent.propertyOperator.createProperty(
            propertyTable="sobelThreshold",
            propertyDefault=70,
            setDefaultOnNetwork=True,
        )
        self.threshold_to_last_used = self.agent.propertyOperator.createProperty(
            propertyTable="threshold_to_last_used_size",
            propertyDefault=25,
            setDefaultOnNetwork=True,
        )
        self.bin_size_pixels = self.agent.propertyOperator.createProperty(
            propertyTable="binning_size_pixels",
            propertyDefault=30,
            setDefaultOnNetwork=True,
        )
        self.min_edge_height = self.agent.propertyOperator.createProperty(
            propertyTable="min_vertical_edge_height",
            propertyDefault=250,  # Minimum height in pixels for a valid edge
            setDefaultOnNetwork=True,
        )
        self.distanceMemoryFrames = self.agent.propertyOperator.createProperty(
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

    def rescaleWidth(self, value):
        return value * self.shape[1] / self.TUNEDWIDTH

    def rescaleHeight(self, value):
        return value * self.shape[0] / self.TUNEDHEIGHT

    def isColorBased(self):
        return False  # uses april tags so b/w frame

    def align(self, inputFrame, draw):
        frame = inputFrame
        if not self.checkFrame(frame):
            # we assume if its not a b/w frame (eg checkframe false), that it means its a cv2 bgr and to change to b/w
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.currentFrameCnt += 1

        newLeftDistance = None
        newRightDistance = None

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

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

                if draw:
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
                if draw:
                    cv2.drawContours(edge_viz, [biggest], -1, (0, 0, 255), 2)

                mid = x + w / 2

                isLeftEdge = mid < frame.shape[1] / 2

                distMid = int(abs(mid - frame.shape[1] / 2))

                if isLeftEdge:
                    newLeftDistance = distMid
                else:
                    newRightDistance = distMid

                if draw:
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

        if draw:
            cv2.putText(
                frame,
                f"L: {selectedLeft}px, R: {selectedRight}px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return selectedLeft, selectedRight
