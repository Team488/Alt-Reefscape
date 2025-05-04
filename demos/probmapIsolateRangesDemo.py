import traceback
from mapinternals.probmap import ProbMap
import random
import cv2

# test sizes all in cm
robotSizeX = 71
robotSizeY = 96
objSize = 35
fieldX = 1000
fieldY = 1600
res = 1  # cm

wX = int(fieldX / 3)
wY = int(fieldY / 3)

# values other than field x,y not used in this demo
fieldMap = ProbMap(
    fieldX, fieldY, res, objSize, objSize, robotSizeX, robotSizeY
)  # Width x Height at 1 cm resolution

isdown = False
lastX = -1
lastY = -1


def __isolateRangeHighest(x, y) -> None:
    (px, py, prob) = fieldMap.getHighestObjectWithinRange(x, y, wX, wY)
    (objMap, robtMap) = fieldMap.getHeatMaps()
    cv2.rectangle(
        objMap,
        (int(x - wX / 2), int(y - wY / 2)),
        (int(x + wX / 2), int(y + wY / 2)),
        (255, 255, 255),
        2,
    )
    print(f"px{px} py{py} prob{prob}")
    if prob > 0:
        cv2.putText(objMap, f"prob{prob}", (x, y), 1, 1, (255, 255, 255))
        cv2.circle(objMap, (px, py), int(6 - prob), (255, 0, 0), 2)
    else:
        cv2.putText(objMap, "No detections in region", (x, y), 1, 1, (255, 255, 255))

    cv2.imshow(fieldMap.gameObjWindowName, objMap)


def __isolateRangeCallback(event, x, y, flags, param) -> None:
    global isdown
    if event == cv2.EVENT_LBUTTONDOWN:
        isdown = True
    elif event == cv2.EVENT_LBUTTONUP:
        isdown = False
    if (event == cv2.EVENT_MOUSEMOVE and isdown) or event == cv2.EVENT_LBUTTONDOWN:
        global lastX
        global lastY
        lastX = x
        lastY = y
        __isolateRangeHighest(x, y)


cv2.namedWindow(fieldMap.gameObjWindowName)
cv2.setMouseCallback(fieldMap.gameObjWindowName, __isolateRangeCallback)


def startDemo() -> None:

    for i in range(20000):
        if i % 5 == 0:
            print("here")
            __test_randomization_ranges(
                fieldMap, int(fieldMap.get_shape()[0]), int(fieldMap.get_shape()[1])
            )

        if i % 15 == 0:
            # slowly dissipate
            fieldMap.disspateOverTime(1)  # 1s
        if lastX != -1:
            __isolateRangeHighest(lastX, lastY)
        else:
            # dont want to be drawing twice
            (objMap, robtMap) = fieldMap.getHeatMaps()
            cv2.imshow(fieldMap.gameObjWindowName, objMap)

        k = cv2.waitKey(100) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            fieldMap.clear_maps()
        # fieldMap.clear_map()


def __test_randomization_ranges(map: ProbMap, width, height) -> None:
    # for i in range(1):
    x = random.randrange(0, width)
    y = random.randrange(0, height)
    # obj_size = 36*6 #size*total potential STD #random.randrange(36, 36)
    confidence = (
        random.randrange(65, 95, 1) / 100
    )  # generates a confidence threshold between 0.65 - 0.95
    try:
        map.addCustomObjectDetection(x, y, 100, 100, confidence)  # 1s since last update

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    startDemo()
