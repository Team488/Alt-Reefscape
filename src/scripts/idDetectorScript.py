# Description: This script uses the Doctr library to detect text in an image of a tag. It loads the image, applies a detection model, and visualizes the detected bounding boxes on the image.
# Requirements: ip install python-doctr[torch,viz]@git+https://github.com/mindee/doctr.git
# Import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
import cv2
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from doctr.utils.geometry import detach_scores

if __name__ == "__main__":
    # Image of tag
    tag_image = cv2.imread("ID1.png")
    tag = open("ID1.png", "rb").read()

    # Helper function to convert relative coordinates to absolute pixel values
    def _to_absolute(geom, img_shape: tuple[int, int]) -> list[list[int]]:
        h, w = img_shape
        if len(geom) == 2:  # Assume straight pages = True -> [[xmin, ymin], [xmax, ymax]]
            (xmin, ymin), (xmax, ymax) = geom
            xmin, xmax = int(round(w * xmin)), int(round(w * xmax))
            ymin, ymax = int(round(h * ymin)), int(round(h * ymax))
            return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        else:  # For polygons, convert each point to absolute coordinates
            return [[int(point[0] * w), int(point[1] * h)] for point in geom]

    # Define the detection predictor
    det_predictor = detection_predictor(
        arch="fast_small",
        pretrained=True,
        assume_straight_pages=True,
        symmetric_pad=True,
        preserve_aspect_ratio=True,
        batch_size=1,
    )  # .cuda().half()  # Uncomment this line if you have a GPU

    # Define the postprocessing parameters (optional)
    det_predictor.model.postprocessor.bin_thresh = 0.3
    det_predictor.model.postprocessor.box_thresh = 0.1

    # Load the document image
    docs = DocumentFile.from_images([tag])
    results = det_predictor(docs)

    for doc, res in zip(docs, results):
        img_shape = (doc.shape[0], doc.shape[1])
        # Detach the probability scores from the results
        detached_coords, prob_scores = detach_scores([res.get("words")])

        for i, coords in enumerate(detached_coords[0]):
            coords = coords.reshape(2, 2).tolist() if coords.shape == (4,) else coords.tolist()

            # Convert relative to absolute pixel coordinates
            points = np.array(_to_absolute(coords, img_shape), dtype=np.int32).reshape((-1, 1, 2))

            # Draw the bounding box on the image
            cv2.polylines(tag_image, [points], isClosed=True, color=(255, 0, 0), thickness=2)

    plt.imshow(cv2.cvtColor(tag_image, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()