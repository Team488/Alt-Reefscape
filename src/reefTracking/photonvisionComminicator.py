import numpy as np
from scipy.spatial.transform import Rotation
from networktables import NetworkTablesInstance
from tools import NtUtils


class PhotonVisionCommunicator:
    def __init__(self, useNetworkTables: bool = True) -> None:
        if useNetworkTables:
            self.net = NetworkTablesInstance.getDefault()
            self.net.initialize("127.0.0.1")
            self.photonvisionTable = self.net.getTable("photonvision")

        else:
            # TODO
            raise NotImplementedError("Only networktables is currently supported!")

    def __affine_matrix_from_quaternion_translation(self, quat, translation):
        # Convert quaternion to rotation matrix
        r = Rotation.from_quat(
            [quat[0], quat[1], quat[2], quat[3]]
        )  # (x, y, z, w) format in scipy
        rot_matrix = r.as_matrix()

        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        translation = np.array(
            [translation[1], translation[2], translation[0]]
        )  # (x,y,z) -> (y,z,x) coordinate systems are so anoyyyyyyying
        T[:3, 3] = translation

        return T

    def getTagPoseAsMatrix(self, photonCameraName):
        network_bytes = (
            self.photonvisionTable.getSubTable(photonCameraName)
            .getEntry("targetPose")
            .getRaw(defaultValue=None)
        )
        if network_bytes:
            unpacked = NtUtils.getTranslation3dFromBytes(network_bytes)
            poseMatrix = self.__affine_matrix_from_quaternion_translation(
                unpacked[1], unpacked[0]
            )
            return poseMatrix

        return None
