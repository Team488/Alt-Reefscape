import math

import cv2
import numpy as np

from Alt.Core.Agents import BindableAgent
from Alt.Cameras.Agents import CameraUsingAgentBase
from Alt.Cameras.Captures import OpenCVCapture



class AlignmentChecker(CameraUsingAgentBase, BindableAgent):
    DEFAULTTHRESH = 10
    PHOTONVISIONMJPGURL = "http://localhost:1181/stream.mjpg"

    @classmethod
    def bind(cls, showFrames: bool, mjpeg_url = PHOTONVISIONMJPGURL):
        cls.__getBindedAgent(
            showFrames=showFrames, mjpeg_url=mjpeg_url
        )

    def __init__(self, showFrames: bool, mjpeg_url = PHOTONVISIONMJPGURL):
        super().__init__(
            capture=OpenCVCapture(name=mjpeg_url, capturePath=mjpeg_url), showFrames=showFrames
        )

    def create(self) -> None:
        super().create()
        self.offAngleProp = self.propertyOperator.createCustomReadOnlyProperty(
            propertyTable="alignmentOffsetErrorRadians",
            propertyValue=-1,
            addBasePrefix=True,
            addOperatorPrefix=False,
        )
        self.offPixProp = self.propertyOperator.createCustomReadOnlyProperty(
            propertyTable="alignmentOffsetErrorPx",
            propertyValue=-1,
            addBasePrefix=True,
            addOperatorPrefix=False,
        )
        self.isCenteredConfidently = self.propertyOperator.createCustomReadOnlyProperty(
            propertyTable="alignedConfidently",
            propertyValue=False,
            addBasePrefix=True,
            addOperatorPrefix=False,
        )
        self.cameraIntrinsics = self.propertyOperator.createReadExistingNetworkValueProperty(
            propertyTable="photonvision.Apriltag_Back_Camera.cameraIntrinsics",
            propertyDefault=None,
        )
        self.threshold_pixels = self.propertyOperator.createProperty(
            propertyTable="threshold_pixels",
            propertyDefault=self.DEFAULTTHRESH,
            setDefaultOnNetwork=True,
        )

    def runPeriodic(self) -> None:
        super().runPeriodic()

        frame = self.latestFrameMain
        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Threshold the image
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # 3. Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Prepare an empty image (same size as frame) to visualize contours
        contour_img = np.zeros_like(frame)

        if contours:
            # Pick the largest contour (assuming that's the main black rectangle)
            largest_contour = max(contours, key=cv2.contourArea)

            # Compute bounding box (x, y, w, h)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate left and right offsets
            left_offset = x
            right_offset = frame.shape[1] - (x + w)
            print(
                f"Left offset: {left_offset:.1f}px, Right offset: {right_offset:.1f}px"
            )

            # 4. Check if it's horizontally centered
            image_center_x = frame.shape[1] / 2
            box_center_x = x + w / 2
            horizontal_offset = abs(image_center_x - box_center_x)

            intr = self.cameraIntrinsics.get()
            if intr is not None and len(intr) > 0:
                intr = np.reshape(intr, (3, 3))
                hres = 800
                vres = 600
                fx = intr[0][0]
                fy = intr[1][1]
                cx = intr[0][2]
                cy = intr[1][2]
                fov_x = 2 * math.atan(hres / 2 * fx)
                fov_per_pix = fov_x / hres

                horizontal_ang_offset = horizontal_offset * fov_per_pix
                self.offAngleProp.set(horizontal_ang_offset)

            self.offPixProp.set(horizontal_offset)

            # Print whether it's centered within 'threshold_pixels'
            if horizontal_offset <= self.threshold_pixels.get():
                self.isCenteredConfidently.set(True)

            else:
                self.isCenteredConfidently.set(False)

            # Draw bounding box on the main frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw all contours in a separate image
            cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)
            # And draw the bounding box there as well
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            self.offPixProp.set(-1)
            self.offAngleProp.set(-1)
            self.isCenteredConfidently.set(False)

    def getDescription(self) -> str:
        return "Looks-At-Camera-April-Tag-Checks-Alignment"

