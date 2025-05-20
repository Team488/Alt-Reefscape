from functools import partial

import cv2
import numpy as np

from Alt.Cameras.Agents import CameraUsingAgentBase
from Alt.Cameras.Captures import OpenCVCapture



class VerticalAlignmentChecker(CameraUsingAgentBase):
    DEFAULTTHRESH = 20  # Default threshold in pixels

    def __init__(self, showFrames: bool, flushTimeMS: int = -1):
        mjpeg_url = "http://localhost:1184/stream.mjpg"
        super().__init__(
            capture=FileCapture(videoFilePath=mjpeg_url, flushTimeMS=flushTimeMS),
            showFrames=showFrames,
        )

    def create(self) -> None:
        super().create()
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
        self.isCenteredConfidently = self.propertyOperator.createCustomReadOnlyProperty(
            propertyTable="verticalAlignedConfidently",
            propertyValue=False,
            addBasePrefix=False,
            addOperatorPrefix=False,
        )
        self.threshold_pixels = self.propertyOperator.createProperty(
            propertyTable="vertical_threshold_pixels",
            propertyDefault=self.DEFAULTTHRESH,
            setDefaultOnNetwork=True,
        )
        self.min_edge_height = self.propertyOperator.createProperty(
            propertyTable="min_vertical_edge_height",
            propertyDefault=50,  # Minimum height in pixels for a valid edge
            setDefaultOnNetwork=True,
        )

    def runPeriodic(self) -> None:
        super().runPeriodic()

        frame = self.latestFrameMain

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Sobel operator to detect vertical edges (x-direction)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        sobel_8u = np.uint8(abs_sobelx / abs_sobelx.max() * 255)

        # Threshold the edge image
        _, thresh = cv2.threshold(sobel_8u, 50, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to enhance vertical edges
        kernel_vertical = np.ones((5, 1), np.uint8)
        vertical_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_vertical)

        # Find contours in the vertical edge image
        contours, _ = cv2.findContours(
            vertical_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Prepare a visualization image
        edge_viz = np.zeros_like(frame)

        min_height = self.min_edge_height.get()
        valid_contours = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter for vertical edges (height > width and minimum height)
            if h > w and h >= min_height:
                valid_contours.append(contour)
                cv2.rectangle(edge_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if valid_contours:
            # Find the largest vertical edge by height
            filtered_contours = [
                contour
                for contour in valid_contours
                if (cv2.boundingRect(contour)[3] > frame.shape[0] / 2)
            ]
            if len(filtered_contours) >= 2:
                filtered_contours = filtered_contours[:2]  # keep max two

            self.rightDistanceProp.set(-1)  # defaults
            self.leftDistanceProp.set(-1)

            leftCenterOffset = None
            rightCenterOffset = None
            distance_diff = -1
            for big_contour in filtered_contours:
                x, y, w, h = cv2.boundingRect(big_contour)
                isRightContour = x > (frame.shape[1] / 2)
                mid = x + w / 2
                if isRightContour:
                    self.rightDistanceProp.set(frame.shape[1] - mid)
                else:
                    self.leftDistanceProp.set(mid)

                # Check if it's centered (distances should be similar)

                distance_diff = abs(mid - frame.shape[1] / 2)

                if isRightContour:
                    rightCenterOffset = distance_diff
                else:
                    leftCenterOffset = distance_diff

                # Draw the largest vertical edge
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Dist from goal {distance_diff}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if leftCenterOffset is not None and rightCenterOffset is not None:
                diff = abs(rightCenterOffset - leftCenterOffset)
                if diff <= self.threshold_pixels.get():
                    self.isCenteredConfidently.set(True)
                else:
                    self.isCenteredConfidently.set(False)

            # Draw all valid vertical edges in the visualization
            cv2.drawContours(edge_viz, valid_contours, -1, (0, 0, 255), 2)

        else:
            self.leftDistanceProp.set(-1)
            self.rightDistanceProp.set(-1)
            self.isCenteredConfidently.set(False)

        # If showing frames is enabled, display the edge visualization
        if self.showFrames:
            cv2.imshow("Vertical Edges", edge_viz)

    def getDescription(self) -> str:
        return "Detects-Vertical-Edges-For-AprilTag-Alignment"


def partialVerticalAlignmentCheck(showFrames: bool = False, flushTimeMS: int = -1):
    return partial(
        VerticalAlignmentChecker, showFrames=showFrames, flushTimeMS=flushTimeMS
    )
