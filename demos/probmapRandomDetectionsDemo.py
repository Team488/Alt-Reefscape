import time

import cv2
from mapinternals import probmap
import random
import traceback
import numpy as np

# randomizes values for stress testing algorithm
def startDemo() -> None:
    # test sizes all in cm
    robotSizeX = 71  # using max dimensions for this of 28 inch
    robotSizeY = 96  # using max dimensions for this of 38 inch

    objSize = 35  # using notes from last year with a outside diameter of 14 inch
    # fieldX = 2743 # roughly 90 ft
    # fieldY = 1676 # roughly 55 ft
    fieldX = 1680
    fieldY = 1000
    res = 2  # cm

    # axis aligned so robot detections will be need to be adjusted for accuracy
    fieldMap = probmap.ProbMap(
        fieldX, fieldY, res, objSize, objSize, robotSizeX, robotSizeY
    )  # Width x Height at 1 cm resolution
    while 1:
        __test_randomization_ranges(
            fieldMap, int(fieldMap.get_shape()[1]), int(fieldMap.get_shape()[0])
        )
        coords = fieldMap.getAllObjectsAboveThreshold(0.4)  # threshold is .4
        objMap = fieldMap.getGameObjectHeatMap()
        fieldMap.disspateOverTime(0.2)
        if coords:
            for coord in coords:
                (px, py, r, prob) = coord
                if prob > 0:
                    # cv2.putText(objMap,f"prob{prob}",(x,y),1,1,(255,255,255))
                    cv2.circle(
                        objMap,
                        (px // fieldMap.resolution, py // fieldMap.resolution),
                        r + 10,
                        (255, 0, 0),
                        2,
                    )
        else:
            cv2.putText(
                objMap,
                "No detections in map",
                (int(fieldX / 2), int(fieldY / 2)),
                1,
                1,
                (255, 255, 255),
            )

        cv2.imshow(fieldMap.gameObjWindowName, objMap)
        # fieldMap.disspateOverTime(1)  # 1s
        # fieldMap.clear_map()
        k = cv2.waitKey(100) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            map.clear_maps()


def __test_randomization_ranges(map: probmap.ProbMap, width, height) -> None:
    x = random.randrange(0, width)
    y = random.randrange(0, height)
    confidence = (
        random.randrange(65, 95, 1) / 100
    )  # generates a confidence threshold between 0.65 - 0.95

    print(f"x{x} y{y} conf{confidence}")
    map.addCustomObjectDetection(x, y, 150, 150, confidence)


if __name__ == "__main__":
    startDemo()
