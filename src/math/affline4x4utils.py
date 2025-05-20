import numpy as np
from scipy.spatial.transform import Rotation

def inverse(matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of a 4x4 affine transformation matrix
    
    Args:
        matrix: 4x4 affine transformation matrix
        
    Returns:
        Inverted 4x4 affine transformation matrix
    """
    rot = matrix[:3, :3]
    rot_inv = np.transpose(rot)
    transf = matrix[:3, 3]
    transf_inv = -rot_inv @ transf

    matrix_inv = np.eye(4)
    matrix_inv[:3, :3] = rot_inv
    matrix_inv[:3, 3] = transf_inv
    return matrix_inv


def extract_angles(matrix: np.ndarray, format: str) -> np.ndarray:
    """
    Extract euler angles from a transformation matrix
    
    Args:
        matrix: Transformation matrix from which to extract angles
        format: Format string for the Euler angle sequence (e.g., 'xyz', 'zyx')
        
    Returns:
        Array of Euler angles in radians
    """
    # Extract the 3x3 rotation matrix
    rot_matrix = matrix[:3, :3]

    euler_angles = Rotation.from_matrix(rot_matrix).as_euler(
        format, degrees=False
    )

    return euler_angles
