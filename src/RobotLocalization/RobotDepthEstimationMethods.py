import math
import cv2
import numpy as np
from Alt.Core import getChildLogger
from Alt.ObjectLocalization.Localization.DepthEstimationMethod import KnownSizeMethod

Sentinel = getChildLogger("Robot_Color_Depth_Method")

class RobotKnownSizeMethod(KnownSizeMethod):
    """ Monocular color camera alternative to depth camera localization for robots
        Uses the numbers on the side of a robots bumper to make a depth estimate 
    """
    __blueRobotHist, _ = np.load("histograms/blueRobotHist.npy")
    __redRobotHist, _ = np.load("histograms/redRobotHist.npy")
    __minPerc = 0.2
    __MAXRATIO = 2 # max deformation of number. With biggest value (can be either h or w) divided by smallest value
    __ROBOTBUMPERHEIGHTCM = 12.7 # very rough estimate as there can be variance between different robots bumper heights
    __KERNELPERAREA = 3500 # will be used to get a kernel size for the opening and closing ops based on image area

    def getObjectSizeCM(self):
        return self.__ROBOTBUMPERHEIGHTCM


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
    def getObjectSizePixels(self, croppedColorBBoxFrame) -> float:
        y, x = croppedColorBBoxFrame.shape[:2]
        if y < 2 or x < 2:
            print("Invalid bbox!")
            return None

        # extracting lower half of frame - assumption bumper is in the bottom portion
        croppedColorBBoxFrame = croppedColorBBoxFrame[croppedColorBBoxFrame.shape[0]//2:, :]

        # Convert to LAB color space
        labFrame = cv2.cvtColor(croppedColorBBoxFrame, cv2.COLOR_BGR2LAB)

        # try backprojecting red and blue histograms and see if one sticks
        # if both fail, isblue will return None
        # processed is the output backprojection
        processed, isBlue = self.__backprojCheck(
            labFrame, self.__redRobotHist, self.__blueRobotHist
        )
        if isBlue is None:
            return None
        
        # use 2n+1 to guarantee that kernel size is odd
        area = x * y
        kSize = 2 * (area//(2 * self.__KERNELPERAREA)) + 1
        k = np.ones((kSize, kSize))

        # Morphological operations for bumper
        bumper_closed = cv2.morphologyEx(
            processed, cv2.MORPH_CLOSE, k, iterations=2
        )
        bumper_opened = cv2.morphologyEx(
            bumper_closed, cv2.MORPH_OPEN, k, iterations=2
        )

        # Find contours
        contours, _ = cv2.findContours(
            bumper_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            Sentinel.warning("Failed to extract a bumper, not enough contour area!")
            return None

        combined_contour = np.concatenate(contours)
        convex_hull = cv2.convexHull(combined_contour)

        # Crop bumper region
        bumperOnlyLab = self.__crop_contours(labFrame, convex_hull)
        # bumperOnly = self.__crop_contours(croppedColorBBoxFrame, convex_hull)

        # Number extraction
        backProjNumbers = self.__backProjWhite(bumperOnlyLab)

        # Morphological operations for numbers
        initalOpen = cv2.morphologyEx(
            backProjNumbers, cv2.MORPH_OPEN, k, iterations=1
        )

        # we aim to collaplse close numbers together into one blob
        # this way we remove lots of the noise that comes from trying to isolate numbers alone
        close = cv2.morphologyEx(
            initalOpen, cv2.MORPH_CLOSE, k, iterations=2
        )

        _, threshNumbers = cv2.threshold(close, 50, 255, cv2.THRESH_BINARY)
        contoursNumbers, _ = cv2.findContours(
            threshNumbers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # prioritize the biggest number shaped blob
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
                    np.zeros_like(threshNumbers, dtype=np.uint8),
                    sorted_contours,
                    -1,
                    (255),
                    -1,
                )

                final_number_image = cv2.morphologyEx(
                    drawn_final_contours, cv2.MORPH_DILATE, k, iterations=1
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

                    numberWidth, numberHeight = min_area_rect[1]
                    
                    # while it sounds counterintutive, the goal is to get the noisy numbers to join into a blob (for noise reduction)
                    # this means that you will get a rectangular object, and the height should be the smaller one
                    targetheight = min(numberHeight, numberWidth)

                    return targetheight

            else:
                Sentinel.debug("Failed to find acceptable contours!")
        else:
            Sentinel.debug("Failed to extract number from contour!")

        return None
    

    def __crop_contours(self, image, contour):
        """ Keep only inside a specified contour and make the rest black"""
        # Unpack the rectangle properties
        mask = np.zeros_like(image)

        # Draw the convex hull on the mask
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        result = np.zeros_like(image)

        # Copy the region of interest using the mask
        result[mask == 255] = image[mask == 255]
        return result


    def __backProjWhite(self, labImage, threshold=120):
        """ White color backprojection"""

        # return cv2.calcBackProject([bumperOnlyLab],[1,2],whiteNumHist,[0,256,0,256],1)
        L, a, b = cv2.split(labImage)

        # Threshold the L channel to get a binary image
        # Here we assume white has high L values, you might need to adjust the threshold value
        _, white_mask = cv2.threshold(L, threshold, 255, cv2.THRESH_BINARY)

        # kernel = np.ones((5, 5), np.uint8)
        # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        # white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        return white_mask

    
    def __backprojAndThreshFrame(self, frame, histogram):
        """
            returns the backprojected frame with either true for blue for false for red
            if there was no color at all the frame returned will have a corresponding None value
        """
        backProj = cv2.calcBackProject([frame], [1, 2], histogram, [0, 256, 0, 256], 1)
        _, thresh = cv2.threshold(backProj, 50, 255, cv2.THRESH_BINARY)
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
        redBackproj = self.__backprojAndThreshFrame(frame, redHist)
        blueBackproj = self.__backprojAndThreshFrame(frame, blueHist)
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

