import math
from typing import Union, Tuple, List, Sequence, cast
import numpy as np
from tools.Constants import MapConstants, Label
from scipy.spatial.transform import Rotation as R

from numba import njit

# Type aliases for numeric types
Numeric = Union[int, float]
VectorLike = Union[Sequence[Numeric], np.ndarray]


def getDistance(
    detectionX: Numeric, 
    detectionY: Numeric, 
    oldX: Numeric, 
    oldY: Numeric, 
    velX: Numeric, 
    velY: Numeric, 
    timeStepSeconds: Numeric
) -> float:
    """
    Calculate the Euclidean distance between a detection and a predicted position
    
    Args:
        detectionX: X coordinate of the detection
        detectionY: Y coordinate of the detection
        oldX: X coordinate of the current position
        oldY: Y coordinate of the current position
        velX: X component of velocity
        velY: Y component of velocity
        timeStepSeconds: Time step for prediction
        
    Returns:
        Euclidean distance between detection and predicted position
    """
    newPositionX = oldX + velX * timeStepSeconds
    newPositionY = oldY + velY * timeStepSeconds
    dx = detectionX - newPositionX
    dy = detectionY - newPositionY
    dist = np.linalg.norm([dx, dy])  # euclidian
    return cast(float, dist)


def calculateMaxRange(
    vx: Numeric, 
    vy: Numeric, 
    timeStep: Numeric, 
    label: Label
) -> float:
    """
    Calculate the maximum range an object could travel based on its velocity and label
    
    Args:
        vx: X component of velocity
        vy: Y component of velocity
        timeStep: Time step for prediction
        label: Object label type
        
    Returns:
        Maximum possible range the object could travel in the given time step
    """
    velocityComponent = np.linalg.norm([vx * timeStep, vy * timeStep])
    if label is not Label.ROBOT:
        return cast(float, velocityComponent)
    # only considering acceleration if its a robot
    maxAcceler = MapConstants.RobotAcceleration.value
    accelerationComponent = (maxAcceler * timeStep * timeStep) / 2
    return cast(float, velocityComponent + accelerationComponent)


def cosineDistance(vec1: VectorLike, vec2: VectorLike) -> float:
    """
    Calculate the cosine distance between two vectors
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine distance between the vectors (dot product divided by product of magnitudes)
    """
    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitudes (norms)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # Calculate cosine similarity
    return cast(float, dot_product / (norm_vec1 * norm_vec2))


@njit
def inverse4x4Affline(matrix: np.ndarray) -> np.ndarray:
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


def extract_pitch_yaw_roll(matrix: np.ndarray, format: str) -> np.ndarray:
    """
    Extract pitch, yaw, and roll angles from a transformation matrix
    
    Args:
        matrix: Transformation matrix from which to extract angles
        format: Format string for the Euler angle sequence (e.g., 'xyz', 'zyx')
        
    Returns:
        Array of Euler angles in radians
    """
    # Extract the 3x3 rotation matrix
    rot_matrix = matrix[:3, :3]

    # Convert to Euler angles (ZYX order: Yaw-Pitch-Roll)
    euler_angles = R.from_matrix(rot_matrix).as_euler(
        format, degrees=False
    )  # should be pitch yaw roll

    return euler_angles
