import logging
import math
import cv2
import numpy as np
from mapinternals.NumberMapper import NumberMapper
from tools.Constants import CameraIntrinsics, InferenceMode, Label, ObjectReferences
from Core.ConfigOperator import staticLoad
from Core import getChildLogger


Sentinel = getChildLogger("Position_Estimator")


class PositionEstimator:
    def __init__(self, isSimulationMode=False, tryocr=False) -> None:
        self.tryocr = tryocr
        self.numMapper = NumberMapper(["6328"], ["6328"])  # update teams
        if tryocr:
            import pytesseract
            from sys import platform

            if platform == "win32":
                pytesseract.pytesseract.tesseract_cmd = (
                    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
                )
            else:
                try:
                    # if this fails, the pytesseract executable has not been installed
                    pytesseract.image_to_string(np.zeros((40, 40), dtype=np.uint8))
                except Exception:
                    print("To use OCR make sure tesseract is installed on your machine")
                    print(f"{{sudo apt-get install tesseract-ocr}}")
                    exit(0)
            self.pytesseract = pytesseract
        self.__minPerc = 0.005  # minimum percentage of bounding box with bumper color
        # simulation currently only has blue robots
        if isSimulationMode:
            self.__blueRobotHist, _ = staticLoad(
                "histograms/simulationBlueRobotCinematicHist.npy"
            )
        else:
            self.__blueRobotHist, _ = staticLoad("histograms/blueRobotHist.npy")

        self.__redRobotHist, _ = staticLoad("histograms/redRobotHist.npy")
        self.__MAXRATIO = 3  # max ratio between number width/height or vice versa

    """ Extract a rectangular slice of the image, given a bounding box. This is axis aligned"""

    def __crop_image(
        self, image, top_left, bottom_right, safety_margin=0
    ):  # in decimal percentage. Eg 5% margin -> 0.05
        x1, y1 = top_left
        x2, y2 = bottom_right
        if safety_margin != 0:
            xMax, yMax = image.shape[1], image.shape[0]
            x1 = int(np.clip(x1 * (1 + safety_margin), 0, xMax))
            x2 = int(np.clip(x2 * (1 + safety_margin), 0, xMax))
            y1 = int(np.clip(y1 * (1 + safety_margin), 0, yMax))
            y2 = int(np.clip(y2 * (1 + safety_margin), 0, yMax))

        cropped_image = image[y1:y2, x1:x2]

        return cropped_image

    """ Keep only inside a specified contour and make the rest black"""

    def __crop_contours(self, image, contour):
        # Unpack the rectangle properties
        mask = np.zeros_like(image)

        # Draw the convex hull on the mask
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        result = np.zeros_like(image)

        # Copy the region of interest using the mask
        result[mask == 255] = image[mask == 255]
        return result

    """ White color backprojection"""

    def __backProjWhite(self, labImage, threshold=120):
        # return cv2.calcBackProject([bumperOnlyLab],[1,2],whiteNumHist,[0,256,0,256],1)
        L, a, b = cv2.split(labImage)

        # Threshold the L channel to get a binary image
        # Here we assume white has high L values, you might need to adjust the threshold value
        _, white_mask = cv2.threshold(L, threshold, 255, cv2.THRESH_BINARY)

        # kernel = np.ones((5, 5), np.uint8)
        # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        return white_mask

    # returns the backprojected frame with either true for blue for false for red
    # if there was no color at all the frame returned will have a corresponding None value
    def __backprojAndThreshFrame(self, frame, histogram, isBlue):
        backProj = cv2.calcBackProject([frame], [1, 2], histogram, [0, 256, 0, 256], 1)
        # cv2.imshow(f"backprojb b?:{isBlue}",backProj)
        _, thresh = cv2.threshold(backProj, 50, 255, cv2.THRESH_BINARY)
        # cv2.imshow(f"thresh b?:{isBlue}",thresh)

        return thresh

    def __getMajorityWhite(self, thresholded_image):
        # Count the number of white pixels (255)
        num_white_pixels = np.sum(thresholded_image == 255)

        # Calculate the total number of pixels
        total_pixels = thresholded_image.size

        # Calculate the percentage of white pixels
        white_percentage = num_white_pixels / total_pixels
        return white_percentage

    """ Checks a frame for two backprojections. Either a blue or red bumper. If there is enough of either color, then its a sucess and we return the backprojected value. Else a fail"""

    def __backprojCheck(self, frame, redHist, blueHist):
        redBackproj = self.__backprojAndThreshFrame(frame, redHist, False)
        blueBackproj = self.__backprojAndThreshFrame(frame, blueHist, True)
        # cv2.imshow("Blue backproj",blueBackproj)
        redPerc = self.__getMajorityWhite(redBackproj)
        bluePerc = self.__getMajorityWhite(blueBackproj)

        if redPerc > bluePerc:
            if redPerc > self.__minPerc:
                print("Red suceess")
                return (redBackproj, False)
            else:
                # failed minimum percentage
                print("Red fail", redPerc)
                return (None, None)
        else:
            # blue greater
            if bluePerc > self.__minPerc:
                print("blue sucess")
                return (blueBackproj, True)
            else:
                print("blue fail", bluePerc)
                return (None, None)

    """ Calculates distance from object assuming we know real size and pixel size. Dimensions out are whatever known size dimensions are
        Its as follows
        knownSize(whatever length dim) * focallength(px)/currentsizePixels(px))

        Output dim is whatever length dim
    """

    def __calculateDistance(self, knownSize, currentSizePixels, focalLengthPixels):
        # todo find calibrated values for other cams
        return (knownSize * focalLengthPixels) / (currentSizePixels)

    """ calculates angle change per pixel, and multiplies by number of pixels off you are. Dimensions are whatever fov per pixel dimensions are

    """

    def __calcBearing(self, fov, res, pixelDiff):
        fovperPixel = fov / res
        return -pixelDiff * fovperPixel

    """
        This is a multistep process to estimate the height of a robot bumper. TLDR use number on the side of bumper to estimate height

        A couple of assumptions at play here, but they seem to always be the case
        Assuming that the number on the side of the bumper is the same height as the bumper
        Assuming that there is a number on every side of the bumper
        Assuming that the detection bounding box ecompasses the whole robot, and thus the bumper will be in the lower half of the clipped bounding box provided

        Steps
        #1 Isolate bottom half of cropped out robot detection.
        #2 try to backproject a red or blue histogram to isolate the bumper. NOTE: it is critical that histograms be updated in every different lighting condiditon (Todo make a better histogram tool)
        #3 if you pass and there is significant red or blue in the frame to indicate a bumper, now we "cut out" the bumper which is found using contours and convex hull.
        #4 we threshold for a white value because the numbers are white. (Possible todo: Use histograms instead)
        #5 we try to isolate numbers from the thresholded white value. We check to see if any "numbers" found are actually number shaped with some simple ratio checks
        #6 if we have found a proper number, we get its height and take that as the bumper height

    """

    def __estimateRobotBumperHeight(self, croppedframe) -> tuple[float, bool]:
        y, x = croppedframe.shape[:2]
        if y < 2 or x < 2:
            print("Invalid bbox!")
            return None

        # Cutting the frame: bumper is in the bottom portion
        croppedframe = self.__crop_image(croppedframe, (0, int(y / 2)), (x, y))

        # Convert to LAB color space
        labFrame = cv2.cvtColor(croppedframe, cv2.COLOR_BGR2LAB)
        processed, isBlue = self.__backprojCheck(
            labFrame, self.__redRobotHist, self.__blueRobotHist
        )
        if isBlue is None:
            return None
        small_frame_threshold = 3000
        # Adjust kernel size and iterations based on frame size
        small_frame = (y * x) < small_frame_threshold
        bumperKernel = np.ones((1, 1) if small_frame else (2, 2), np.uint8)
        iterations_close = 1 if small_frame else 2
        iterations_open = 1 if small_frame else 2

        # Morphological operations for bumper
        bumper_closed = cv2.morphologyEx(
            processed, cv2.MORPH_CLOSE, bumperKernel, iterations=iterations_close
        )
        bumper_opened = cv2.morphologyEx(
            bumper_closed, cv2.MORPH_OPEN, bumperKernel, iterations=iterations_open
        )

        # Find contours
        contours, _ = cv2.findContours(
            bumper_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            logging.warning("Failed to extract a bumper, not enough contour area!")
            return None

        combined_contour = np.concatenate(contours)
        convex_hull = cv2.convexHull(combined_contour)

        # Crop bumper region
        bumperOnlyLab = self.__crop_contours(labFrame, convex_hull)
        bumperOnly = self.__crop_contours(croppedframe, convex_hull)

        # Number extraction
        kerneltwobytwo = np.ones((1, 1) if small_frame else (2, 2), np.uint8)
        backProjNumbers = self.__backProjWhite(bumperOnlyLab)

        # Morphological operations for numbers
        initalOpen = cv2.morphologyEx(
            backProjNumbers, cv2.MORPH_OPEN, kerneltwobytwo, iterations=iterations_open
        )
        nums = ""
        if self.tryocr:
            nums = self.pytesseract.image_to_string(backProjNumbers)

        close = cv2.morphologyEx(
            initalOpen, cv2.MORPH_CLOSE, kerneltwobytwo, iterations=iterations_close
        )
        final_open = cv2.morphologyEx(
            close, cv2.MORPH_OPEN, kerneltwobytwo, iterations=1
        )

        _, threshNumbers = cv2.threshold(final_open, 50, 255, cv2.THRESH_BINARY)
        contoursNumbers, _ = cv2.findContours(
            threshNumbers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if contoursNumbers:
            acceptableContours = []
            for contour in contoursNumbers:
                min_area_rect = cv2.minAreaRect(contour)
                numberWidth, numberHeight = min_area_rect[1]
                if numberHeight == 0 or numberWidth == 0:
                    continue
                ratio = max(numberWidth, numberHeight) / min(numberWidth, numberHeight)
                if ratio < self.__MAXRATIO:
                    acceptableContours.append(contour)

            if acceptableContours:
                sorted_contours = sorted(
                    acceptableContours, key=cv2.contourArea, reverse=True
                )
                drawn_final_contours = cv2.drawContours(
                    np.zeros_like(final_open, dtype=np.uint8),
                    sorted_contours,
                    -1,
                    (255),
                    -1,
                )
                final_number_image = cv2.morphologyEx(
                    drawn_final_contours, cv2.MORPH_DILATE, kerneltwobytwo, iterations=1
                )
                largest_acceptable_contour = max(
                    cv2.findContours(
                        final_number_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )[0],
                    key=cv2.contourArea,
                )

                if largest_acceptable_contour is not None:
                    min_area_rect = cv2.minAreaRect(largest_acceptable_contour)
                    box = cv2.boxPoints(min_area_rect)
                    box = np.int0(box)
                    cv2.drawContours(final_open, [box], 0, (255), 2)
                    numberWidth, numberHeight = min_area_rect[1]
                    targetheight = min(numberHeight, numberWidth)
                    print(
                        f"HEIGHT----------------------------{targetheight}----------------------------"
                    )
                    return targetheight, isBlue, nums

            else:
                logging.warning("Failed to find acceptable contours!")
        else:
            logging.warning("Failed to extract number from contour!")

        return None

    """ Method takes in a frame, where you have already run your model. It crops out bounding boxes for each robot detection and runs a height estimation.
        If it does not fail to estimate a height, it then takes that information, along with a calculated bearing to estimate a relative position
       """

    def __estimateRelativeRobotPosition(
        self, frame, boundingBox, cameraIntrinsics: CameraIntrinsics
    ) -> tuple[float, float]:
        x1, y1, x2, y2 = boundingBox
        w = x2 - x1
        h = y2 - y1
        midW = int(w / 2)
        midH = int(h / 2)
        centerX = x1 + midW
        croppedImg = self.__crop_image(frame, (x1, y1), (x2, y2), safety_margin=0.07)
        est = self.__estimateRobotBumperHeight(croppedImg)
        if est is not None:
            (estimatedHeight, isBlue, nums) = est
            distance = self.__calculateDistance(
                ObjectReferences.BUMPERHEIGHT.getMeasurementCm(),
                estimatedHeight,
                cameraIntrinsics.getFy(),
            )
            bearing = self.__calcBearing(
                cameraIntrinsics.getHFovRad(),
                cameraIntrinsics.getHres(),
                int(centerX - cameraIntrinsics.getCx()),
            )
            print(f"{distance=} {est=} {bearing=}")
            estCoords = self.componentizeHDistAndBearing(distance, bearing)

            return estCoords
        return None

    def __estimateRelativeCoralPosition(
        self, frame, boundingBox, cameraIntrinsics: CameraIntrinsics
    ) -> tuple[float, float]:
        # TODO!
        return None

    """ This current method estimates the position of a note, by using the same method as a robot. However it is slightly simplified, as we can take avantage of the circular nature of a note
        By taking the width of a note (or the max of w and h to cover the case when its vertical), we can find a pretty much exact value for the size of the note in pixels. Given we know the
        exact size of a note, we can then use this to estimate distance.

    """

    def __estimateRelativeNotePosition(
        self, frame, boundingBox, cameraIntrinsics: CameraIntrinsics
    ) -> tuple[float, float]:
        x1, y1, x2, y2 = boundingBox
        w = x2 - x1
        h = y2 - y1
        midW = int(w / 2)
        # midH = int(h / 2)
        centerX = x1 + midW
        objectSize = max(w, h)
        distance = self.__calculateDistance(
            ObjectReferences.NOTE.getMeasurementCm(),
            objectSize,
            cameraIntrinsics.getFx(),
        )
        bearing = self.__calcBearing(
            cameraIntrinsics.getHFovRad(),
            cameraIntrinsics.getHres(),
            int(centerX - cameraIntrinsics.getCx()),
        )
        print(f"{bearing=}")
        estCoords = self.componentizeHDistAndBearing(distance, bearing)
        return estCoords

    def __estimateRelativeAlgaePosition(
        self, frame, boundingBox, cameraIntrinsics: CameraIntrinsics
    ) -> tuple[float, float]:
        x1, y1, x2, y2 = boundingBox
        w = x2 - x1
        h = y2 - y1
        midW = int(w / 2)
        # midH = int(h / 2)
        centerX = x1 + midW
        objectSize = max(w, h)
        distance = self.__calculateDistance(
            ObjectReferences.ALGAEDIAMETER.getMeasurementCm(),
            objectSize,
            cameraIntrinsics.getFx(),
        )
        bearing = self.__calcBearing(
            cameraIntrinsics.getHFovRad(),
            cameraIntrinsics.getHres(),
            int(centerX - cameraIntrinsics.getCx()),
        )
        print(f"{bearing=}")
        estCoords = self.componentizeHDistAndBearing(distance, bearing)
        return estCoords

    def __estimateRelativePosition(
        self,
        class_idx: int,
        frame: np.ndarray,
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
            return self.__estimateRelativeRobotPosition(frame, bbox, cameraIntrinsics)
        if label == Label.NOTE:
            return self.__estimateRelativeNotePosition(frame, bbox, cameraIntrinsics)
        if label == Label.ALGAE:
            return self.__estimateRelativeAlgaePosition(frame, bbox, cameraIntrinsics)
        Sentinel.warning(
            f"Label: {str(label)} is not supported for position estimation!"
        )
        return None

    def estimateDetectionPositions(
        self,
        frame,
        labledResults,
        cameraIntrinsics: CameraIntrinsics,
        inferenceMode: InferenceMode,
    ):
        estimatesOut = []
        for result in labledResults:
            class_idx = result[3]
            bbox = result[1]
            estimate = self.__estimateRelativePosition(
                class_idx, frame, bbox, cameraIntrinsics, inferenceMode
            )

            if estimate is not None:
                estimatesOut.append(
                    [result[0], estimate, result[2], class_idx, result[4]]
                )  # replace local bbox with estimated position
            # else we dont include this result
            # todo keep a metric of failed estimations
            else:
                print("Failed estimation")

        return estimatesOut
    
    from ..Constants.Inference import Object
    def __estimateRelativePosition(
        self, frame : np.ndarray, boundingBox, cameraIntrinsics: CameraIntrinsics, label : Object
    ) -> tuple[float, float]:
        x1, y1, x2, y2 = boundingBox
        w = x2 - x1
        h = y2 - y1
        midW = int(w / 2)
        midH = int(h / 2)
        centerX = x1 + midW
        croppedImg = self.__crop_image(frame, (x1, y1), (x2, y2), safety_margin=0.07)
        
        depth = label.depthMethod.getDepthEstimateCM(croppedImg, cameraIntrinsics)
        if not depth:
            return None
        
        bearing = self.__calcBearing(
            cameraIntrinsics.getHFovRad(),
            cameraIntrinsics.getHres(),
            int(centerX - cameraIntrinsics.getCx()),
        )

        Sentinel.debug(f"{depth=} {bearing=}")

        estCoords = self.componentizeHDistAndBearing(depth, bearing)
        return estCoords


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


