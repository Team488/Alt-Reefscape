from enum import Enum
import numpy as np
from Units import Types, Conversions


class ReefBranches(Enum):
    @staticmethod
    def getDefaultLengthType():
        return Types.Length.IN

    L2L = (0, "L2-L", np.array([-6.756, -19.707, 2.608]))
    L2R = (1, "L2-R", np.array([6.754, -19.707, 2.563]))
    L3L = (2, "L3-L", np.array([-6.639, -35.606, 2.628]))
    L3R = (3, "L3-R", np.array([6.637, -35.606, 2.583]))
    L4L = (4, "L4-L", np.array([-6.470, -58.4175, 0.921]))
    L4R = (5, "L4-R", np.array([6.468, -58.4175, 0.876]))

    @property
    def branchid(self) -> int:
        return self.value[0]

    @property
    def branchname(self) -> str:
        return self.value[1]

    def getAprilTagOffset(self, units: Types.Length = Types.Length.CM) -> np.ndarray:
        # The original value is a numpy array, so we need to handle it carefully
        result = Conversions.convertLength(
            self.value[2], self.getDefaultLengthType(), units
        )

        return result

    @classmethod
    def getByID(cls, branchid: int):
        for branch in cls:
            if branch.branchid == branchid:
                return branch

        return None

