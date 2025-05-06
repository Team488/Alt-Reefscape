import numpy as np
from Alt.ObjectLocalization.Constants.InferenceC import Object
from Alt.ObjectLocalization.Localization.DepthEstimationMethod import KnownSizeMethod, DepthCameraMethod


class RobotEstimationMethod(DepthCameraMethod):
    def getDepthValueCM(self, croppedDepthBBoxFrameCM):
        diffThresh = 10  # cm
        deltas = np.diff(croppedDepthBBoxFrameCM, axis=0)

        startY = croppedDepthBBoxFrameCM.shape[0] - 2  # One less due to np.diff
        centerX = croppedDepthBBoxFrameCM.shape[1] // 2
        centerW = 25 #px

        while startY >= 0:
            deltaRow = deltas[startY, centerX-centerW:centerX+centerW]

            if np.nanmax(np.abs(deltaRow)) < diffThresh:
                depthRow = croppedDepthBBoxFrameCM[startY + 1]  # +1 because of diff shift
                minDepth = np.nanmin(depthRow)
                return minDepth if not np.isnan(minDepth) else None
            startY -= 1

        return None

    
ROBOT = Object("Robot", (75, 75), RobotEstimationMethod)

class AlgaeEstimationMethod(KnownSizeMethod):
    def getObjectSizeCM(self):
        return 41
    
    def getObjectSizePixels(self, croppedColorBBoxFrame):
        # model will detect whole algea surrounded in bounding box. Thus we can just maximize the w/h of the box
        return max(croppedColorBBoxFrame.shape[:2])

ALGAE = Object("algae", (41, 41), AlgaeEstimationMethod)
CORAL = Object("coral", (30, 12), None)