from collections import defaultdict
import cv2
import io
from PIL import Image
import numpy as np
from abstract.AlignmentProvider import AlignmentProvider
from Core import PropertyOperator, getChildLogger
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from doctr.utils.geometry import detach_scores

Sentinel = getChildLogger("DocTr_Alignment_Provider")


class DocTrAlignmentProvider(AlignmentProvider):
    def __init__(self):
        super().__init__()

    def create(self):
        self.initalizerDetector()

    def initalizerDetector(self):
        self.det_predictor = detection_predictor(
            arch="fast_small",
            pretrained=True,
            assume_straight_pages=True,
            symmetric_pad=True,
            preserve_aspect_ratio=True,
            batch_size=1,
        )  # .cuda().half()  # Uncomment this line if you have a GPU

        # Define the postprocessing parameters (optional)
        self.det_predictor.model.postprocessor.bin_thresh = 0.3
        self.det_predictor.model.postprocessor.box_thresh = 0.1

    def isColorBased(self):
        return False  # uses april tags so b/w frame

    def align(self, inputFrame, draw):
        frame = inputFrame  # move og ref of input frame to draw on original
        if not self.checkFrame(frame):
            # we assume if its not a b/w frame (eg checkframe false), that it means its a cv2 bgr and to change to b/w
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pil_image = Image.fromarray(frame)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")

        docs = DocumentFile.from_images([img_byte_arr.getvalue()])
        results = self.det_predictor(docs)
        left = None
        right = None

        for doc, res in zip(docs, results):
            img_shape = (doc.shape[0], doc.shape[1])
            # Detach the probability scores from the results
            detached_coords, prob_scores = detach_scores([res.get("words")])

            for i, coords in enumerate(detached_coords[0]):
                coords = (
                    coords.reshape(2, 2).tolist()
                    if coords.shape == (4,)
                    else coords.tolist()
                )

                # Convert relative to absolute pixel coordinates
                points = np.array(
                    self._to_absolute(coords, img_shape), dtype=np.int32
                ).reshape((-1, 1, 2))

                if draw:
                    cv2.polylines(
                        inputFrame,
                        [points],
                        isClosed=True,
                        color=(255, 0, 0),
                        thickness=2,
                    )

                if len(points) > 0:
                    left = left if left is not None else points[0][0]
                    right = right if right is not None else points[0][0]

                vals = [point[0] for point in points]
                vals.append(left)
                vals.append(right)
                left = min(vals)
                right = max(vals)

        return left, right

    # Helper function to convert relative coordinates to absolute pixel values
    def _to_absolute(self, geom, img_shape: tuple[int, int]) -> list[list[int]]:
        h, w = img_shape
        if (
            len(geom) == 2
        ):  # Assume straight pages = True -> [[xmin, ymin], [xmax, ymax]]
            (xmin, ymin), (xmax, ymax) = geom
            xmin, xmax = int(round(w * xmin)), int(round(w * xmax))
            ymin, ymax = int(round(h * ymin)), int(round(h * ymax))
            return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        else:  # For polygons, convert each point to absolute coordinates
            return [[int(point[0] * w), int(point[1] * h)] for point in geom]
