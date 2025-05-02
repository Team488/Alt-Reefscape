import numpy as np
from enum import Enum
from Units import Types, Conversions
from Constants.constants import TEAM
from typing import Optional, Tuple
from scipy.spatial.transform import Rotation


Position2D = Tuple[float, float]
Position3D = Tuple[float, float, float]
RotationAngles = Tuple[float, float]

class ATLocations(Enum):
    """
    AprilTag locations with ID, (x, y, z) coordinates in inches, and (yaw, pitch) rotations in degrees.
    """

    @staticmethod
    def getDefaultLengthType():
        return Types.Length.IN

    @staticmethod
    def getDefaultRotationType():
        return Types.Rotation.Deg

    TAG_1 = ((1), (657.37, 25.80, 58.50), (126, 0), None, None)
    TAG_2 = ((2), (657.37, 291.20, 58.50), (234, 0), None, None)
    TAG_3 = ((3), (455.15, 317.15, 51.25), (270, 0), None, None)
    TAG_4 = ((4), (365.20, 241.64, 73.54), (0, 30), None, None)
    TAG_5 = ((5), (365.20, 75.39, 73.54), (0, 30), None, None)
    TAG_6 = ((6), (530.49, 130.17, 12.13), (300, 0), TEAM.RED, 1)
    TAG_7 = ((7), (546.87, 158.50, 12.13), (0, 0), TEAM.RED, 2)
    TAG_8 = ((8), (530.49, 186.83, 12.13), (60, 0), TEAM.RED, 1)
    TAG_9 = ((9), (497.77, 186.83, 12.13), (120, 0), TEAM.RED, 2)
    TAG_10 = ((10), (481.39, 158.50, 12.13), (180, 0), TEAM.RED, 1)
    TAG_11 = ((11), (497.77, 130.17, 12.13), (240, 0), TEAM.RED, 2)
    TAG_12 = ((12), (33.51, 25.80, 58.50), (54, 0), None, None)
    TAG_13 = ((13), (33.51, 291.20, 58.50), (306, 0), None, None)
    TAG_14 = ((14), (325.68, 241.64, 73.54), (180, 30), None, None)
    TAG_15 = ((15), (325.68, 75.39, 73.54), (180, 30), None, None)
    TAG_16 = ((16), (235.73, -0.15, 51.25), (90, 0), None, None)
    TAG_17 = ((17), (160.39, 130.17, 12.13), (240, 0), TEAM.BLUE, 1)
    TAG_18 = ((18), (144.00, 158.50, 12.13), (180, 0), TEAM.BLUE, 2)
    TAG_19 = ((19), (160.39, 186.83, 12.13), (120, 0), TEAM.BLUE, 1)
    TAG_20 = ((20), (193.10, 186.83, 12.13), (60, 0), TEAM.BLUE, 2)
    TAG_21 = ((21), (209.49, 158.50, 12.13), (0, 0), TEAM.BLUE, 1)
    TAG_22 = ((22), (193.10, 130.17, 12.13), (300, 0), TEAM.BLUE, 2)

    @property
    def id(self):
        return self.value[0]

    @property
    def position(self):
        return self.value[1]

    @property
    def rotation(self):
        return self.value[2]

    @property
    def team(self):
        return self.value[3]

    @property
    def blockingAlgaeLevel(self):
        return self.value[4]

    @classmethod
    def get_by_id(cls, tag_id):
        """Retrieve an ATLocation by its ID."""
        for tag in cls:
            if tag.id == tag_id:
                return tag
        return None

    @classmethod
    def get_pose_by_id(
        cls,
        tag_id: int,
        length: Types.Length = Types.Length.CM,
        rotation_type: Types.Rotation = Types.Rotation.Rad,
    ) -> Tuple[Position3D, RotationAngles]:
        """Retrieve the position and rotation for a given tag ID."""
        tag = cls.get_by_id(tag_id)

        if tag is None:
            return None

        # Get position
        position = Conversions.convertLength(
            tag.position, cls.getDefaultLengthType(), length
        )

        # Get rotation
        rotation = Conversions.convertRotation(
            tag.rotation, cls.getDefaultRotationType(), rotation_type
        )

        return position, rotation

    @classmethod
    def getPoseAfflineMatrix(
        cls, tag_id: int, units: Types.Length = Types.Length.CM
    ) -> Optional[np.ndarray]:
        """Returns a 4x4 affine matrix for the tag pose"""
        pose = cls.get_pose_by_id(tag_id, length=units)
        if pose is None:
            return None

        translation, rotation_angles = pose
        # Make sure the translation is a 3D position
        if not isinstance(translation, tuple) or len(translation) != 3:
            return None

        # Convert rotation angles to a rotation matrix
        rotMatrix = Rotation.from_euler(
            "ZY", rotation_angles, degrees=False
        ).as_matrix()

        m = np.eye(4)
        m[:3, :3] = rotMatrix
        m[:3, 3] = translation
        return m

    @classmethod
    def getReefBasedIds(cls, team: Optional[TEAM] = None) -> list[int]:
        if not team:
            return ATLocations.getReefBasedIds(TEAM.BLUE) + ATLocations.getReefBasedIds(
                TEAM.RED
            )

        ids = []
        for tag in cls:
            if tag.team == team:
                ids.append(tag.id)

        return ids

    @classmethod
    def getAlgaeLevel(cls, atId):
        tag = cls.get_by_id(atId)

        if tag is None:
            return None

        return tag.blockingAlgaeLevel

    @classmethod
    def getBlockedBranchIdxs(cls, atId):
        level = cls.getAlgaeLevel(atId)

        if level is None:
            return None

        if level == 1:
            return 0, 1, 2, 3
        if level == 2:
            return 2, 3, 4, 5
