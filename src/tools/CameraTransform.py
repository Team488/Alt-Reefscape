import math
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


