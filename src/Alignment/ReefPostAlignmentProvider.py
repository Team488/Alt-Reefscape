import time
import cv2
import numpy as np
from .AlignmentProvider import AlignmentProvider
from Core import getChildLogger
from Core.ConfigOperator import staticLoad
from tools import generalutils
from sklearn.cluster import DBSCAN

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

        self.agent

        # Adjustable processing parameters
        self.sobelksize = self.propertyOperator.createProperty("sobelksize", 3)
        self.threshold_pre = self.propertyOperator.createProperty(
            "backproject_threshold", 150
        )
        self.threshold_sobel = self.propertyOperator.createProperty("Sobel_Thresh", 150)
        self.roi_fraction = self.propertyOperator.createProperty("ROI_Fraction", 0.5)
        self.min_line_length = self.propertyOperator.createProperty(
            "Min_Line_Length", 50
        )
        self.max_line_length = self.propertyOperator.createProperty(
            "Max_Line_Length", 60
        )
        self.angle_range = self.propertyOperator.createProperty("Angle_Range", 10)
        self.xdiff_threshold = self.propertyOperator.createProperty(
            "clustering_x_diff(pixels)", 25
        )
        self.angle_threshold = self.propertyOperator.createProperty(
            "clustering_angle_diff(degrees)", 5
        )
        self.min_lines_per_cluster = self.propertyOperator.createProperty(
            "Min_Lines_Per_Cluster", 3
        )
        self.cluster_size_tolerance = self.propertyOperator.createProperty(
            "Cluster_Size_Tolerance", 10
        )
        self.erosionIter = self.propertyOperator.createProperty(
            "Hughlines_pre_erosion", 10
        )
        self.min_cluster_width = self.propertyOperator.createProperty(
            "Min_Cluster_Width", 20
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
            # draw roi
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

        # Sobel edge detection
        sobel_x = cv2.Sobel(
            threshold_pre, cv2.CV_64F, 1, 0, ksize=self.sobelksize.get()
        )
        sobel_abs = np.uint8(np.absolute(sobel_x) / np.absolute(sobel_x).max() * 255)
        _, edge_thresh = cv2.threshold(
            sobel_abs, self.threshold_sobel.get(), 255, cv2.THRESH_BINARY
        )

        # Morphological closing
        vertical_kernel = np.ones((5, 1), np.uint8)
        vertical_edges = cv2.morphologyEx(
            cv2.erode(edge_thresh, np.ones((2, 2)), iterations=self.erosionIter.get()),
            cv2.MORPH_CLOSE,
            vertical_kernel,
        )

        # Detect vertical lines using Hough Transform
        lines = cv2.HoughLinesP(
            vertical_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=self.min_line_length.get(),
            maxLineGap=10,
        )

        if lines is None:
            return None, None

        # Extract x-coordinates and angles

        line_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            avg_x = (x1 + x2) / 2
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

            # Only consider near-vertical lines
            if (90 - self.angle_range.get()) < angle < (90 + self.angle_range.get()):
                line_data.append((avg_x, angle, line))

                if draw:
                    cv2.line(
                        frame,
                        (x1 + left_bound, y1),
                        (x2 + left_bound, y2),
                        (255, 255, 255),
                        2,
                    )

        if not line_data:
            return None, None

        # Apply DBSCAN clustering on X-coordinates
        x_coords = []
        angles = []
        for x, angle, _ in line_data:
            # normalize angle and x coords to range defined by their respective maximums
            x_coords.append(x / self.xdiff_threshold.get())
            angles.append(angle / self.angle_threshold.get())

        features = np.column_stack((x_coords, angles))

        # Apply DBSCAN clustering on (x, angle)
        dbscan = DBSCAN(eps=1, min_samples=self.min_lines_per_cluster.get())
        labels = dbscan.fit_predict(features)

        # Organize clusters
        clusters = {}
        for i, label in enumerate(labels):
            if label != -1:  # Ignore noise
                clusters.setdefault(label, []).append(line_data[i])

        # Find best cluster: widest and closest to center
        best_cluster = None
        max_width = 0
        for cluster_lines in clusters.values():
            x_positions = [line[2][0][0] for line in cluster_lines] + [
                line[2][0][2] for line in cluster_lines
            ]
            leftmost = min(x_positions) + left_bound
            rightmost = max(x_positions) + left_bound
            width = rightmost - leftmost
            center_dist = abs((leftmost + rightmost) / 2 - half_width)

            if draw:
                cv2.rectangle(
                    frame, (leftmost, 0), (rightmost, frame.shape[0]), (0, 0, 255), 3
                )

            if width >= self.min_cluster_width.get() and width > max_width:
                max_width = width
                best_cluster = (leftmost, rightmost, center_dist)

        # Draw best cluster
        if best_cluster and draw:
            cv2.rectangle(
                frame,
                (best_cluster[0], 0),
                (best_cluster[1], frame.shape[0]),
                (0, 255, 0),
                3,
            )
            cv2.putText(
                frame,
                "SELECTED",
                (best_cluster[0], 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        return tuple(map(float, best_cluster[:2])) if best_cluster else (None, None)
