import math
import numpy as np
import cv2
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
        self.obstacles, _ = staticLoad("obstacleMap.npy")
        self.NUMSIMULATEDSTATES = numStates
        self.SINGLESTATELEN = 4
        self.STATELEN = self.NUMSIMULATEDSTATES * self.SINGLESTATELEN
        self.MEASUREMENTLEN = 2
        self.fieldX = fieldX
        self.fieldY = fieldY
        self.maxDist = np.linalg.norm((self.fieldX, self.fieldY))
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

            # noise = np.random.normal(0, 1e-40, size=2)
            noise = (0, 0)
            new_vx = vel_x + noise[0]
            new_vy = vel_y + noise[1]

            # Update particle state
            x[idx : idx + self.SINGLESTATELEN] = new_x, new_y, new_vx, new_vy

        self.baseUKF.P += 1e-20
        return x

    # Check if a position is valid (within bounds and not an obstacle)
    def is_valid_position(self, x, y, robotHeight) -> bool:
        if x < 0 or x > self.fieldX or y < 0 or y > self.fieldY:
            return False  # Outside the bounds
        if self.obstacles[int(y), int(x)] <= robotHeight:
            return False  # Collides with an obstacle
        return True

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

    def pseudo_measurement_update(
        self, pseudo_measurement_noise=1e-3, robotHeight=35
    ) -> None:
        sigmas = self.baseUKF.sigmas_f
        h_sigmas = np.array([self.pseudo_measurement(s, robotHeight) for s in sigmas])
        h_sigmas = np.vstack(h_sigmas)
        # Predicted pseudo-measurement
        z_pred, S = unscented_transform(h_sigmas, self.baseUKF.Wm, self.baseUKF.Wc)
        # print(f"{S=}")
        S += pseudo_measurement_noise

        # Cross-covariance
        Pxz = np.zeros((self.STATELEN, 1))
        for i in range(len(sigmas)):
            Pxz += self.baseUKF.Wc[i] * np.outer(
                sigmas[i] - self.baseUKF.x, h_sigmas[i] - z_pred
            )

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)
        z_pseudo = 0  # Target pseudo-measurement
        self.baseUKF.x += K @ (z_pseudo - z_pred)
        self.baseUKF.P -= K @ S @ K.T

    def pseudo_measurement(self, state, robotHeight):
        penalty = 0
        for i in range(self.NUMSIMULATEDSTATES):
            idx = i * self.SINGLESTATELEN
            x, y, vx, vy = state[idx : idx + self.SINGLESTATELEN]
            new_x = x + vx * self.dt
            new_y = y + vy * self.dt
            if not self.is_valid_position(new_x, new_y, robotHeight):
                penalty += np.linalg.norm([new_x - x, new_y - y])
        print(penalty)
        return np.array([penalty])

    def predict_and_update(self, measurements, robotHeight):
        self.baseUKF.predict()
        self.pseudo_measurement_update(robotHeight=robotHeight)
        measurement = np.array(measurements)
        self.baseUKF.update(measurement)
        return self.baseUKF.x
