import random
import sys
import time
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import demos.utils as demoUtils
from mapinternals.probmap import ProbMap
import cv2
from Core.Central import Central
from pathplanning.PathGenerator import PathGenerator

central = Central()
mapSizeX = central.objectmap.internalHeight
mapSizeY = central.objectmap.internalWidth
print(f"mx{mapSizeX} my{mapSizeY} ")
pathgenerator = PathGenerator(central)
isMouseDownL = False
isMouseDownR = False
needUpdate = True
curLocation = (mapSizeX // 2, mapSizeY // 2)  # start in center


def getInterestingRandomTarget(maxX, maxY, currentX, currentY):
    # im sure this is the smaaartest beest way of doing this
    sigmaX = maxX / 5
    sigmaY = maxY / 5
    gaussianX = random.gauss(currentX, sigmaX)
    gaussianY = random.gauss(currentY, sigmaY)
    # return gaussianX,gaussianY
    flipPointX = (
        currentX + (maxX - currentX) / 2 if gaussianX > currentX else currentX / 2
    )
    flipPointY = (
        currentY + (maxY - currentY) / 2 if gaussianY > currentY else currentY / 2
    )
    retX = flipPointX + (flipPointX - gaussianX)
    retY = flipPointY + (flipPointY - gaussianY)
    retX = min(maxX - 1, max(0, retX))
    retY = min(maxY - 1, max(0, retY))
    return int(retX), int(retY)


def mouseDownCallback(event, x, y, flags, param) -> None:
    global isMouseDownL
    global needUpdate
    rX, rY = demoUtils.rescaleCoordsUp(x, y, central.objectmap)
    if event == cv2.EVENT_LBUTTONDOWN:
        isMouseDownL = True
        #  print("clicked at ", x," ", y)
        central.objectmap.addCustomRobotDetection(
            rX, rY, 400, 400, 1
        )  # adding as a 75% probability
        needUpdate = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if isMouseDownL:
            #   print("dragged at ", x," ", y)
            central.objectmap.addCustomRobotDetection(
                rX, rY, 400, 400, 1
            )  # adding as a 75% probability
            needUpdate = True

    elif event == cv2.EVENT_LBUTTONUP:
        isMouseDownL = False
    elif event == cv2.EVENT_MBUTTONDOWN:
        global curLocation
        curLocation = (x, y)
        needUpdate = True


def n(x) -> None:
    global needUpdate
    needUpdate = True


def getNextMovementVector(currentLocation, path, currentPathIndex):
    vector = None
    if path is not None and len(path) > currentPathIndex:
        vector = path[currentPathIndex]
    return vector


def startDemo() -> None:
    global needUpdate
    global curLocation
    trackbarName = "Min Height in CM"
    cv2.namedWindow(central.objectmap.gameObjWindowName)
    cv2.createTrackbar(trackbarName, central.objectmap.gameObjWindowName, 10, 255, n)
    cv2.setMouseCallback(central.objectmap.gameObjWindowName, mouseDownCallback)
    nextMovementVector = None
    currentPathIndex = 1

    randomTarget = getInterestingRandomTarget(
        mapSizeX, mapSizeY, curLocation[0], curLocation[1]
    )
    print("random target", randomTarget)
    print("our location", curLocation)
    while True:
        if curLocation == randomTarget:
            randomTarget = getInterestingRandomTarget(
                mapSizeX, mapSizeY, curLocation[0], curLocation[1]
            )
            needUpdate = True

        if needUpdate:
            currentPath = pathgenerator.generate(
                curLocation,
                randomTarget,
                minHeightCm=cv2.getTrackbarPos(
                    trackbarName, central.objectmap.gameObjWindowName
                ),
                customObstacleMap=(255 - central.objectmap.getRobotHeatMap()),
                reducePoints=True,
            )
            nextMovementVector = getNextMovementVector(curLocation, currentPath, 1)
            currentPathIndex = 2

            needUpdate = False

        if nextMovementVector is not None:
            curLocation = tuple(
                map(
                    int,
                    np.add(
                        curLocation,
                        demoUtils.getRealisticMoveVector(
                            curLocation, nextMovementVector, 20
                        ),
                    ),
                )
            )
            nextMovementVector = getNextMovementVector(
                curLocation, currentPath, currentPathIndex
            )
            currentPathIndex += 1

        robotMap = central.objectmap.getRobotHeatMap()
        w, h = robotMap.shape
        display_frame = cv2.merge((central.objectmap.getGameObjectHeatMap(), robotMap))
        cv2.circle(display_frame, curLocation, 10, (0, 255, 0), -1)
        cv2.circle(display_frame, randomTarget, 10, (255, 0, 0), -1)

        if currentPath is not None:
            for p in currentPath:
                cv2.circle(display_frame, (int(p[0]), int(p[1])), 2, (255, 255, 0), -1)

        k = cv2.waitKey(100) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            central.objectmap.clear_maps()
            needUpdate = True
        if k == ord("r"):
            randomTarget = getInterestingRandomTarget(
                mapSizeX, mapSizeY, curLocation[0], curLocation[1]
            )
            needUpdate = True
        cv2.imshow(central.objectmap.gameObjWindowName, display_frame)


if __name__ == "__main__":
    startDemo()
