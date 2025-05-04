import numpy as np
import cv2
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


def __addFieldBoundsAsObstacles(
    obstacles, fieldX, fieldY, fieldObstacleDepth=10
) -> None:
    # add obstacles to represent field bounds
    # offset field bounds so its visible
    topRightCorner = (fieldObstacleDepth, fieldObstacleDepth)
    topLeftCorner = (fieldX + fieldObstacleDepth, fieldObstacleDepth)
    bottomRightCorner = (fieldX + fieldObstacleDepth, fieldY + fieldObstacleDepth)
    bottomLeftCorner = (fieldObstacleDepth, fieldY + fieldObstacleDepth)
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
        points = (firstCorner, secondCorner, firstCornerShifted, secondCornerShifted)
        max_point = max(points, key=lambda p: (p[0], p[1]))
        min_point = min(points, key=lambda p: (p[0], p[1]))
        obstacles.append((max_point, min_point))


def __getLine(oldX, oldY, newX, newY) -> tuple[float, float]:
    if oldX == newX:
        return float("inf"), 0
    m = (newY - oldY) / (newX - oldX)
    b = newY - m * newX
    return m, b


def __getXvalue(y, m, b):
    if m == 0:
        return float("inf")
    return (y - b) / m


def __getYvalue(x, m, b):
    return m * x + b


def __getPossibleCollisionSides(oldX, oldY, obstacle) -> tuple[int, int]:
    ((topX, topY), (botX, botY)) = obstacle
    possibleX = topX if oldX > topX else botX
    possibleY = topY if oldY > topY else botY
    return possibleX, possibleY


def __isWithin(oldDim, newDim, topDim, bottomDim):
    topMovement = oldDim if oldDim > newDim else newDim
    bottomMovement = oldDim if oldDim < newDim else newDim
    # handle cases where a point is within first
    if (bottomDim <= topMovement <= topDim) or (bottomDim <= bottomMovement <= topDim):
        return True
    # now check if the old dim and new dim cross these sides
    return topMovement >= topDim and bottomMovement <= bottomDim


def __adjustCollisionToClosestSide(
    oldX, oldY, newX, newY, obstacle
) -> tuple[float, float]:
    collisionPoint = None
    ((topX, topY), (botX, botY)) = obstacle

    # Get line from points
    m, b = __getLine(oldX, oldY, newX, newY)
    print(f"m{m} b{b}")
    # Find the x, y coordinates of the side that it could collide into
    possibleX, possibleY = __getPossibleCollisionSides(oldX, oldY, obstacle)

    # Plug into line equation to get other point in the line, if we have x, then find y or vice versa
    YforPossibleX = __getYvalue(possibleX, m, b)
    XforPossibleY = __getXvalue(possibleY, m, b)

    # Check if this found point is where we collide
    if botY <= YforPossibleX <= topY:
        collisionPoint = (possibleX, YforPossibleX)
    elif botX <= XforPossibleY <= topX:
        collisionPoint = (XforPossibleY, possibleY)
    return collisionPoint


def redrawScene(frame, oldX, oldY, newX, newY, obstacles) -> None:
    cv2.arrowedLine(frame, (oldX, oldY), (newX, newY), (0, 255, 0), 2)

    # Check for obstacle avoidance
    for obstacle in obstacles:
        ((topX, topY), (botX, botY)) = obstacle
        print(f"oX{oldX} oY{oldY} nX{newX} nY{newY}")
        if __isWithin(oldX, newX, topX, botX) and __isWithin(oldY, newY, topY, botY):
            collisionPoint = __adjustCollisionToClosestSide(
                oldX, oldY, newX, newY, obstacle
            )
            if collisionPoint is not None:
                adjustedX, adjustedY = collisionPoint
                cv2.circle(frame, (int(adjustedX), int(adjustedY)), 6, (255, 0, 0), -1)
                break
            else:
                print("Hit edge")
        else:
            print("no collision")
    for obstacle in obstacles:
        cv2.rectangle(frame, obstacle[0], obstacle[1], (0, 0, 255), 2)
    cv2.imshow("frame", frame)


def getNewFrame(fieldX, fieldY):
    return np.zeros((fieldX, fieldY, 3), dtype=np.int8)


def startDemo() -> None:
    # Example usage:
    obstacles = [((100, 100), (50, 50))]
    fieldX = 200
    fieldY = 200
    fieldObstacleDepth = 15
    __addFieldBoundsAsObstacles(obstacles, fieldX, fieldY, fieldObstacleDepth)

    frame = getNewFrame(
        fieldX + 2 * fieldObstacleDepth, fieldY + 2 * fieldObstacleDepth
    )

    lastX1 = int(fieldX / 1.3)
    lastY1 = int(fieldY / 1.3)
    lastX2 = int(fieldX / 2.4)
    lastY2 = int(fieldY / 2.4)

    def updateX1(val) -> None:
        frame = getNewFrame(
            fieldX + 2 * fieldObstacleDepth, fieldY + 2 * fieldObstacleDepth
        )
        global lastX1
        global lastY1
        global lastY2
        global lastX2
        global obstacles

        lastX1 = val
        redrawScene(frame, lastX1, lastY1, lastX2, lastY2, obstacles)

    def updateY1(val) -> None:
        frame = getNewFrame(
            fieldX + 2 * fieldObstacleDepth, fieldY + 2 * fieldObstacleDepth
        )
        global lastX1
        global lastY1
        global lastY2
        global lastX2
        global obstacles

        lastY1 = val
        redrawScene(frame, lastX1, lastY1, lastX2, lastY2, obstacles)

    def updateX2(val) -> None:
        frame = getNewFrame(
            fieldX + 2 * fieldObstacleDepth, fieldY + 2 * fieldObstacleDepth
        )
        global lastX1
        global lastY1
        global lastY2
        global lastX2
        global obstacles

        lastX2 = val
        redrawScene(frame, lastX1, lastY1, lastX2, lastY2, obstacles)

    def updateY2(val) -> None:
        frame = getNewFrame(
            fieldX + 2 * fieldObstacleDepth, fieldY + 2 * fieldObstacleDepth
        )
        global lastX1
        global lastY1
        global lastY2
        global lastX2
        global obstacles

        lastY2 = val
        redrawScene(frame, lastX1, lastY1, lastX2, lastY2, obstacles)

    cv2.namedWindow("testWin")
    cv2.createTrackbar("X1", "testWin", 0, fieldX + 2 * fieldObstacleDepth, updateX1)
    cv2.createTrackbar("Y1", "testWin", 0, fieldY + 2 * fieldObstacleDepth, updateY1)
    cv2.createTrackbar("X2", "testWin", 0, fieldX + 2 * fieldObstacleDepth, updateX2)
    cv2.createTrackbar("Y2", "testWin", 0, fieldY + 2 * fieldObstacleDepth, updateY2)

    # Example prediction and update
    measurements = [60, 60]  # Example measurements
    while True:
        frame = getNewFrame(
            fieldX + 2 * fieldObstacleDepth, fieldY + 2 * fieldObstacleDepth
        )
        redrawScene(frame, lastX1, lastY1, lastX2, lastY2, obstacles)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    startDemo()
