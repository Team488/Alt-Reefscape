import time
import cv2
import numpy as np
from abstract.AlignmentProvider import AlignmentProvider
from Core import getChildLogger
from Core.ConfigOperator import staticLoad
from tools import generalutils

logger = getChildLogger("ReefPostAlignment")


class ReefPostAlignmentProvider(AlignmentProvider):
    def __init__(self):
        super().__init__()

    def create(self):
        """Initialize alignment parameters and load the reference histogram."""
        super().create()

        # Load precomputed color histogram
        self.hist, self.mtime = staticLoad(
            "assets/histograms/reef_post_hist.npy", isRelativeToSource=True
        )

        # Adjustable processing parameters
        self.solidity = self.propertyOperator.createProperty("solidity", 0.8)
        self.threshold_pre = self.propertyOperator.createProperty(
            "backproject_threshold", 30
        )
        self.roi_fraction = self.propertyOperator.createProperty("ROI_Fraction", 0.5)
        self.ksize = self.propertyOperator.createProperty("ksize", 7)
        self.erosionIter = self.propertyOperator.createProperty(
            "Hughlines_pre_erosion", 2
        )

        # Read-only property for histogram update time
        self.propertyOperator.createReadOnlyProperty(
            "Histogram_Update_Time", generalutils.getTimeStr(time.localtime(self.mtime))
        )

    def isColorBased(self):
        """Indicate that alignment is color-based."""
        return True

    def align(self, frame, draw):
        if not self.checkFrame(frame):
            raise ValueError("The frame is not a color frame!")

        # Define region of interest (ROI)
        half_width = frame.shape[1] // 2
        roi_width = int(half_width * self.roi_fraction.get())
        left_bound, right_bound = half_width - roi_width, half_width + roi_width
        roi_frame = frame[:, left_bound:right_bound]

        if draw:
            # Draw ROI
            w = 8
            cv2.line(
                frame,
                (left_bound - w, 0),
                (left_bound - w, frame.shape[0]),
                (0, 0, 0),
                w,
            )
            cv2.line(
                frame,
                (right_bound + w, 0),
                (right_bound + w, frame.shape[0]),
                (0, 0, 0),
                w,
            )

        # Convert to LAB color space and apply back projection
        lab = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2LAB)
        back_proj = cv2.calcBackProject([lab], [1, 2], self.hist, [0, 256, 0, 256], 1)

        _, threshold_pre = cv2.threshold(
            back_proj, self.threshold_pre.get(), 255, cv2.THRESH_BINARY
        )

        eroded = cv2.erode(
            threshold_pre,
            np.ones((self.ksize.get(), self.ksize.get()), np.uint8),
            iterations=self.erosionIter.get(),
        )

        # Find contours
        contours, _ = cv2.findContours(
            eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            frame, [cnt + (left_bound, 0) for cnt in contours], -1, (255, 255, 255), 1
        )

        # Filter based on solidity and use rotated rectangles
        largest_blob = None
        max_width = 0

        bigmask = np.zeros_like(threshold_pre, dtype=np.uint8)

        for contour in contours:
            rotated_rect = cv2.minAreaRect(contour)  # Get rotated rectangle
            (_, (w, h), _) = rotated_rect
            totalArea = w * h
            box = cv2.boxPoints(rotated_rect)
            box = np.int0(box)  # Convert to int

            mask = np.zeros_like(threshold_pre, dtype=np.uint8)
            cv2.fillPoly(mask, [box], (255))

            maskArea = np.count_nonzero(cv2.bitwise_and(mask, threshold_pre))

            bigmask = cv2.bitwise_or(bigmask, mask)

            width = min(rotated_rect[1])  # Get width and height of rotated rect

            if totalArea > 0:
                solidity = (
                    maskArea / totalArea
                )  # Solidity close to 1 means fewer branches
                if draw:
                    cv2.putText(
                        eroded,
                        f"{solidity:.1f}",
                        (
                            int(rotated_rect[0][0] + rotated_rect[1][0]),
                            int(rotated_rect[0][1]),
                        ),
                        1,
                        1,
                        (255),
                        1,
                    )

                # Ensure blob is wide, not on the side, and has a high solidity
                if width > max_width and solidity > self.solidity.get():
                    max_width = width
                    largest_blob = box

        leftright = (None, None)
        # Draw the selected blob as a rotated rectangle
        if largest_blob is not None:
            centroid = np.mean(largest_blob, axis=0)

            leftright = (
                int(centroid[0] + left_bound),
                int(centroid[0] + left_bound),
            )  # since we are only taking the centroid, left and right will be the x

            if draw:
                cv2.drawContours(
                    frame, [largest_blob + (left_bound, 0)], -1, (0, 255, 0), 2
                )

        return leftright
