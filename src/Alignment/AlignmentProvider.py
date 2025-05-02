from abc import ABC, abstractmethod

import numpy as np

from Core.PropertyOperator import PropertyOperator


class AlignmentProvider(ABC):
    """Alignment providers will take in a frame and give left right offset values from the center of the frame\n
    All alignment providers assume that you are centering on the camera, and will provide left/right values as such\n
    NOTE: If the alignment provider is color based (Override the method for isColorBased()), the provider will expect a color frame
    """

    def _inject(self, propertyOperator: PropertyOperator):
        self.propertyOperator = propertyOperator

    @abstractmethod
    def __init__(self) -> None:
        """Initalize the alignment provider here"""
        pass

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def isColorBased(self) -> bool:
        """Returns whether this alignment provider is color based or not"""
        pass

    @abstractmethod
    def align(self, frame: np.ndarray, showFrames: bool) -> tuple[int, int]:
        """Takes in a frame (color is isColorBased() is true), and returns you left,right offsets from the center of the frame
        If a side is not visible, it will return -1 for that side

        Input:
            cv2 frame
        Return:
            left, right pixel offsets from the center
        """
        pass

    def shutDown(self) -> None:
        """Optional Perform alignment provider shutdown here"""
        pass

    def checkFrame(self, frame: np.ndarray) -> bool:
        """Helper method to check if a frame is valid given if the alignment checker is color based or not"""
        numChannels = frame.shape[2]

        isColor = numChannels > 1
        isAlignmentColor = self.isColorBased()

        return isColor == isAlignmentColor
