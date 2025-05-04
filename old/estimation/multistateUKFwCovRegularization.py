import math
import numpy as np
import cv2
import heapq
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman.unscented_transform import unscented_transform
from Core.ConfigOperator import staticLoad
from tools.Constants import MapConstants


class MultistateUkf:
    """Obstacles are expected to be in x,y format"""

    def __init__(
        self,
        numStates,
        fieldX=MapConstants.fieldWidth.value,
        fieldY=MapConstants.fieldHeight.value,
    ) -> None:
        # "Constants"
        self.NUMSIMULATEDSTATES = numStates
        self.SINGLESTATELEN = 4
        self.STATELEN = self.NUMSIMULATEDSTATES * self.SINGLESTATELEN
        self.MEASUREMENTLEN = 2
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.maxDist = np.linalg.norm((self.fieldX, self.fieldY))
        self.fixedRobotHeight = 35

        print("Loading precomputed nearest positions.... this may take a second")
        self.obstacles, _ = staticLoad("obstacleMap.npy")
        self.obstacles_nearest = self.get_nearest_valid_points(self.obstacles)

        # Parameters
        self.dt = 1 / 15  # Time step

        # Covariance matrices
        self.P_initial = np.eye(self.STATELEN) * 0.01
        # self.Q = np.eye(self.STATELEN) * 0.3  # Process noise covariance
        self.Q = np.diag([0.01, 0.01, 0.1, 0.1] * self.NUMSIMULATEDSTATES)
        self.R = np.eye(self.MEASUREMENTLEN) * 0.02  # Measurement noise covariance

        # Sigma points
        self.points = MerweScaledSigmaPoints(
            self.STATELEN, alpha=1e-3, beta=2.0, kappa=-1
        )

        # UKF initialization
        self.baseUKF = UnscentedKalmanFilter(
            dim_x=self.STATELEN,
            dim_z=self.MEASUREMENTLEN,
            fx=self.fx,
            hx=self.hx,
            dt=self.dt,
            points=self.points,
        )
        # Initial state
        self.set_state(0, 0, 0, 0)  # x, y, vx, vy
        self.baseUKF.P = self.P_initial
        self.baseUKF.Q = self.Q
        self.baseUKF.R = self.R

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

    def reset_P(self) -> None:
        self.baseUKF.P = np.eye(self.STATELEN)

    def set_state(self, x, y, vx, vy) -> None:
        newState = [x, y, vx, vy] * self.NUMSIMULATEDSTATES
        self.baseUKF.x = newState

    # State transition function
    def fx(self, x, dt):
        for i in range(self.NUMSIMULATEDSTATES):
            idx = i * self.SINGLESTATELEN
            old_x, old_y, vel_x, vel_y = x[idx : idx + self.SINGLESTATELEN]

            # New state prediction
            new_x = old_x + vel_x * dt
            new_y = old_y + vel_y * dt

            noise = np.random.normal(0, 1e-40, size=2)
            new_vx = vel_x + noise[0]
            new_vy = vel_y + noise[1]

            new_x = np.clip(new_x, 0, self.fieldX - 1) + np.random.normal(0, 1e-20)
            new_y = np.clip(new_y, 0, self.fieldY - 1) + np.random.normal(0, 1e-20)

            if self.obstacles[int(new_y), int(new_x)] <= self.fixedRobotHeight:
                nearestcoord = self.obstacles_nearest[int(new_y)][int(new_x)]
                new_x, new_y = nearestcoord + np.random.normal(0, 1e-10, size=2)
                new_vx, new_vy = (0, 0)

            # Update particle state
            x[idx : idx + self.SINGLESTATELEN] = new_x, new_y, new_vx, new_vy

        self.baseUKF.P = (self.baseUKF.P + self.baseUKF.P.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(self.baseUKF.P)
        eigenvalues = np.maximum(eigenvalues, 0)
        self.baseUKF.P = (
            eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        )  # cov regularization

        return x

    def hx(self, x):
        # Calculate the mean position (x, y) across all particles
        mean_x = np.mean(
            [x[i * self.SINGLESTATELEN] for i in range(self.NUMSIMULATEDSTATES)]
        )
        mean_y = np.mean(
            [x[i * self.SINGLESTATELEN + 1] for i in range(self.NUMSIMULATEDSTATES)]
        )
        return np.array([mean_x, mean_y])  # Return the mean as the measurement

    def getMeanEstimate(self):
        return self.hx(self.baseUKF.x)

    def predict_and_update(self, measurements):
        self.baseUKF.predict()
        measurement = np.array(measurements)
        self.baseUKF.update(measurement)
        return self.baseUKF.x
