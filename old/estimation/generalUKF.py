from typing import Callable, Any
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints


class generalUKF:
    def __init__(
        self,
        initalStateVector,
        measurementDim,
        stateTransitionFunction: Callable[[Any, float], Any],
        measurementFunction: Callable[[Any], Any],
        timeStepS=0.1,
    ) -> None:
        # Parameters
        self.dt = timeStepS
        self.stateTransitionFunction = stateTransitionFunction
        self.measurementFunction = measurementFunction

        # Initial state
        self.Dim = len(initalStateVector)
        self.measurementDim = measurementDim
        self.x_initial = np.array(
            initalStateVector
        )  # Example initial state (position_x, position_y, velocity_x, velocity_y)

        # Covariance matrices
        self.P_initial = np.eye(self.Dim)
        self.Q = np.eye(self.Dim) * 0.01  # Process noise covariance
        self.R = np.eye(self.measurementDim) * 0.01  # Measurement noise covariance

        # Sigma points
        self.points = MerweScaledSigmaPoints(
            self.Dim, alpha=0.5, beta=2.0, kappa=3 - self.Dim
        )

        # UKF initialization
        self.baseUKF = UnscentedKalmanFilter(
            dim_x=self.Dim,
            dim_z=self.measurementDim,
            fx=stateTransitionFunction,
            hx=measurementFunction,
            dt=self.dt,
            points=self.points,
        )
        self.baseUKF.x = self.x_initial
        self.baseUKF.P = self.P_initial
        self.baseUKF.Q = self.Q
        self.baseUKF.R = self.R

    # Example prediction and update steps
    def predict_and_update(self, measurements):
        self.baseUKF.predict()

        measurement = np.array(measurements)
        self.baseUKF.update(measurement)

        return self.baseUKF.x

    def getMeasurement(self):
        return self.measurementFunction(self.baseUKF.x)
