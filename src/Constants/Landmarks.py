from enum import Enum

from Alt.Core.Units import Types, Conversions

class MapConstants(Enum):
    @staticmethod
    def getDefaultLengthType():
        return Types.Length.CM

    @staticmethod
    def getDefaultRotationType():
        return Types.Rotation.Deg

    fieldWidth = 1755  # 54' 3" in cm
    fieldHeight = 805  # 26' 3" in cm

    robotWidth = 75  # cm
    robotHeight = 75  # cm assuming square robot with max frame perimiter of 300
    gameObjectWidth = 35  # cm
    gameObjectHeight = 35  # cm

    b_reef_center = (448.93, 402.59)  # cm
    r_reef_center = (1305.8902, 402.59)  # cm

    reefRadius = 83.185  # cm

    coral_inner_diameter = 10.16  # cm
    coral_outer_diameter = 11.43  # cm

    coral_width = 30.16  # cm

    def getCM(self) -> float:
        return self.value

    def getLength(self, lengthType: Types.Length = Types.Length.CM) -> float:
        result = Conversions.convertLength(
            self.getCM(), fromType=self.getDefaultLengthType(), toType=lengthType
        )
        return result
