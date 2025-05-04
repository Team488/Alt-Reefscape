import math
import numpy as np
import cv2
import heapq
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman.unscented_transform import unscented_transform
from tools.Constants import MapConstants
from Core.ConfigOperator import staticLoad


class PrecomputeNearest:
    """Obstacles are expected to be in x,y format"""

    def __init__(
        self,
        fieldX=MapConstants.fieldWidth.value,
        fieldY=MapConstants.fieldHeight.value,
    ) -> None:
        # "Constants"
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.fixedRobotHeight = 35

        print("Loading precomputed nearest positions.... this may take a second")
        self.obstacles, _ = staticLoad("obstacleMap.npy")
        self.obstacles_nearest = self.get_nearest_valid_points(self.obstacles)

    def get_nearest_valid_points(self, obstacleMap):
        row, col = obstacleMap.shape
        minDistances = [[1000000 for _ in range(col)] for _ in range(row)]
        nearest_valid_points = [
            [
                self.get_nearest_valid_point(i, j, obstacleMap, minDistances)
                for j in range(col)
            ]
            for i in range(row)
        ]
        return nearest_valid_points

    def get_nearest_valid_point(self, i, j, obstacleMap, minDistances):
        nearest = []
        heapq.heappush(
            nearest,
            (
                1000000,
                0,
                i,
                j,
            ),
        )
        visited = set()
        while nearest:
            (_, distance_from_root, i_n, j_n) = heapq.heappop(nearest)
            # print(f"{i_n=} {j_n=} {distance_from_root=}")
            if (i_n, j_n) not in visited:
                if obstacleMap[i_n, j_n] > self.fixedRobotHeight:
                    minDistances[i][j] = distance_from_root
                    return (j_n, i_n)

                visited.add((i_n, j_n))

                # Add neighbors to the heap with boundary checks
                if 0 <= i_n + 1 < self.fieldX:
                    heapq.heappush(
                        nearest,
                        (
                            minDistances[i_n + 1][j_n],
                            distance_from_root + 1,
                            i_n + 1,
                            j_n,
                        ),
                    )
                if 0 <= i_n - 1 < self.fieldX:
                    heapq.heappush(
                        nearest,
                        (
                            minDistances[i_n - 1][j_n],
                            distance_from_root + 1,
                            i_n - 1,
                            j_n,
                        ),
                    )
                if 0 <= j_n + 1 < self.fieldY:
                    heapq.heappush(
                        nearest,
                        (
                            minDistances[i_n][j_n + 1],
                            distance_from_root + 1,
                            i_n,
                            j_n + 1,
                        ),
                    )
                if 0 <= j_n - 1 < self.fieldY:
                    heapq.heappush(
                        nearest,
                        (
                            minDistances[i_n][j_n - 1],
                            distance_from_root + 1,
                            i_n,
                            j_n - 1,
                        ),
                    )

        return (-1, -1)


mouse_position = (0, 0)

# Mouse callback function to update the observation
def mouse_callback(event, x, y, flags, param) -> None:
    global mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_position = np.array([x, y])


t = PrecomputeNearest()
cv2.namedWindow("test")
cv2.setMouseCallback("test", mouse_callback)
while True:
    frame = t.obstacles.copy()
    nearestCoord = t.obstacles_nearest[mouse_position[1]][mouse_position[0]]

    cv2.circle(frame, nearestCoord, (10), (170), -1)
    cv2.circle(frame, mouse_position, (5), (170), -1)

    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
