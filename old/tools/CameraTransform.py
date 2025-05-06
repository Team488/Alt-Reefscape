import math
from typing import Optional, Tuple, List, Any, Union, cast
import numpy as np


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 transformation matrix
    
    Args:
        T: A 4x4 transformation matrix
        
    Returns:
        The inverse transformation matrix
    """
    R = T[:3, :3]  # Rotation matrix
    t = T[:3, 3]  # Translation vector

    R_inv = R.T  # Inverse of rotation matrix (R^-1 = R^T for rotation matrices)
    t_inv = -R_inv @ t  # Inverse translation properly calculated

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv


# Example transformation matrices (for reference, not used in production)
# Robot to target transformation
T_robot_to_target: np.ndarray = np.array(
    [[1, 0, 0, 1.524], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)

# Camera rotation angle in radians
camRot: float = math.radians(179.78)

# Camera to target transformation
T_camera_to_target: np.ndarray = np.array(
    [
        [math.cos(camRot), -math.sin(camRot), 0, 1.19],
        [math.sin(camRot), math.cos(camRot), 0, -0.34],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
