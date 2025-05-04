import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from Alt.Core.Constants.Field import Field

from ..Constants.Kalman import KalmanConstants


class Ukf:
    """Obstacles are expected to be in x,y format"""

    def __init__(
        self,
        field : Field,
    ) -> None:

        self.fieldX = field.getWidth()
        self.fieldY = field.getHeight()
        self.__addFieldBoundsAsObstacles(self.fieldX, self.fieldY)
        
        # Parameters
        self.dt = KalmanConstants.Dt  # Time step

        # Initial state (wont be used)
        self.Dim = 4 # x, y, vx, vy -> len of 4
        self.x_initial = np.array(
            [0, 0, 0, 0] 
        )  

        # Covariance matrices
        self.P_initial = np.eye(4)
        self.Q = np.eye(4) * 0.01  # Process noise covariance
        self.R = np.eye(2) * 0.01  # Measurement noise covariance

        # Sigma points TODO parameterize inputs
        self.points = MerweScaledSigmaPoints(
            self.Dim, alpha=0.5, beta=2.0, kappa=3 - self.Dim
        )

        # UKF initialization (dim z = 2 beccause we only measure x, y)
        self.baseUKF = UnscentedKalmanFilter(
            dim_x=4, dim_z=2, fx=self.fx, hx=self.hx, dt=self.dt, points=self.points
        )
        self.baseUKF.x = self.x_initial
        self.baseUKF.P = self.P_initial
        self.baseUKF.Q = self.Q
        self.baseUKF.R = self.R

    def __addFieldBoundsAsObstacles(
        self, fieldX, fieldY, fieldObstacleDepth=10
    ) -> list:
        obstacles = []
        # add obstacles to represent field bounds
        topRightCorner = (0, 0)
        topLeftCorner = (fieldX, 0)
        bottomRightCorner = (fieldX, fieldY)
        bottomLeftCorner = (0, fieldY)
        corners = (topRightCorner, topLeftCorner, bottomRightCorner, bottomLeftCorner)
        dirsX = [(-1, 0), (1, 1), (1, 0), (-1, -1)]
        dirsY = [(-1, -1), (-1, 0), (1, 1), (1, 0)]
        for i in range(1, 5):
            firstCorner = corners[i - 1]
            secondCorner = corners[i % 4]
            (xShift1, xShift2) = dirsX[i - 1]
            (yShift1, yShift2) = dirsY[i - 1]
            firstCornerShifted = (
                firstCorner[0] + xShift1 * fieldObstacleDepth,
                firstCorner[1] + yShift1 * fieldObstacleDepth,
            )
            secondCornerShifted = (
                secondCorner[0] + xShift2 * fieldObstacleDepth,
                secondCorner[1] + yShift2 * fieldObstacleDepth,
            )
            points = (
                firstCorner,
                secondCorner,
                firstCornerShifted,
                secondCornerShifted,
            )
            max_point = max(points, key=lambda p: (p[0], p[1]))
            min_point = min(points, key=lambda p: (p[0], p[1]))
            obstacles.append((max_point, min_point))
        
        return obstacles

    # State transition function
    def fx(self, x, dt):
        old_x, old_y, vel_x, vel_y = x
        new_x = old_x + vel_x * dt
        new_y = old_y + vel_y * dt
        # # Check for obstacle avoidance
        # for obstacle in self.obstacles:
        #     ((topX, topY), (botX, botY)) = obstacle
        #     # print(obstacle)
        #     if self.__isWithin(old_x, new_x, topX, botX) and self.__isWithin(
        #         old_y, new_y, topY, botY
        #     ):
        #         collisionPoint = self.__adjustCollisionToClosestSide(
        #             old_x, old_y, new_x, new_y, obstacle
        #         )
        #         if collisionPoint is not None:
        #             print(collisionPoint)
        #             adjustedX, adjustedY = collisionPoint
        #             new_x = adjustedX

        #             new_y = adjustedY
        #             break
        #     print("no collision")
        #     print("Current " + str((new_x, new_y)))

        return np.array([new_x, new_y, vel_x, vel_y])

    def __adjustCollisionToClosestSide(
        self, oldX, oldY, newX, newY, obstacle
    ) -> tuple[float, float]:
        collisionPoint = None
        ((topX, topY), (botX, botY)) = obstacle

        # Get line from points
        m, b = self.__getLine(oldX, oldY, newX, newY)
        # Find the x, y coordinates of the side that it could collide into
        possibleX, possibleY = self.__getPossibleCollisionSides(oldX, oldY, obstacle)

        # Plug into line equation to get other point in the line, if we have x, then find y or vice versa
        YforPossibleX = self.__getYvalue(possibleX, m, b)
        XforPossibleY = self.__getXvalue(possibleY, m, b)

        # Check if this found point is where we collide
        # Check if this found point is where we collide
        if botY <= YforPossibleX <= topY:
            collisionPoint = (possibleX, YforPossibleX)
        elif botX <= XforPossibleY <= topX:
            collisionPoint = (XforPossibleY, possibleY)
        return collisionPoint

    def __isWithin(self, oldDim, newDim, topDim, bottomDim):
        topMovement = oldDim if oldDim > newDim else newDim
        bottomMovement = oldDim if oldDim < newDim else newDim
        # handle cases where a point is within first
        if (bottomDim <= topMovement <= topDim) or (
            bottomDim <= bottomMovement <= topDim
        ):
            return True
        # now check if the old dim and new dim cross these sides
        return topMovement >= topDim and bottomMovement <= bottomDim

    def __getPossibleCollisionSides(self, oldX, oldY, obstacle) -> tuple[int, int]:
        ((topX, topY), (botX, botY)) = obstacle
        possibleX = topX if oldX > topX else botX
        possibleY = topY if oldY > topY else botY
        return possibleX, possibleY

    def __getLine(self, oldX, oldY, newX, newY) -> tuple[float, float]:
        if oldX == newX:
            return float("inf"), 0
        m = (newY - oldY) / (newX - oldX)
        b = newY - m * newX
        return m, b

    def __getXvalue(self, y, m, b):
        if m == 0:
            return float("inf")
        return (y - b) / m

    def __getYvalue(self, x, m, b):
        return m * x + b

    # Define the measurement function
    def hx(self, x):
        return np.array([x[0], x[1]])

    # Example prediction and update steps
    def predict_and_update(self, measurements):
        self.baseUKF.predict()
        # print(f"Predicted state: {self.baseUKF.x}")

        # Example measurement update (assuming perfect measurement for demonstration)
        measurement = np.array(measurements)
        self.baseUKF.update(measurement)
        # print(f"Updated state: {self.baseUKF.x}")
        return self.baseUKF.x


# # Example usage:
# obstacles = [((100, 100), (50, 50))]
# fieldX = 200
# fieldY = 200


# ukf = Ukf(obstacles, fieldX, fieldY)

# # Example prediction and update
# measurements = [50, 50]  # Example measurements
# ukf.predict_and_update(measurements)
