import traceback
from mapinternals.probmap import ProbMap
import random
import cv2
import math

# test sizes all in cm
robotSizeX = 71
robotSizeY = 96
objSize = 35
fieldX = 1600  # roughly 90 ft
fieldY = 1000  # roughly 55 ft
res = 2  # cm

wX = int(fieldX / 3)
wY = int(fieldY / 3)

# values other than field x,y not used in this demo
fieldMap = ProbMap(
    fieldX, fieldY, res, objSize, objSize, robotSizeX, robotSizeY
)  # Width x Height at 1 cm resolution

maxRobotSpeed = 60  # cm/s
objectsCollected = 0
lastCollectedX = -1
lastCollectedY = -1


# method also displays stuff so not just getting highest range
def getRangeHighest(x, y):
    global objectsCollected
    global lastCollectedX
    global lastCollectedY
    highest = None
    (px, py, prob) = fieldMap.getHighestObjectWithinRangeT(
        x, y, wX, wY, 0.30
    )  # .30 threshold
    (objMap, robtMap) = fieldMap.getHeatMaps()
    cv2.rectangle(
        objMap,
        (
            int(x - wX / 2),
            int(y - wY / 2),
        ),
        (
            int(x + wX / 2),
            int(y + wY / 2),
        ),
        (255),
        2,
    )
    print(f"px{px} py{py} prob{prob}")
    if prob > 0:
        cv2.putText(objMap, f"prob{prob}", (x, y), 1, 1, (255))
        # cv2.circle(objMap,(px,py),int(6-prob),(255,0,0),2)
        cv2.circle(objMap, (px, py), 5, (255, 0, 0), 2)
        highest = (px, py)
    else:
        cv2.putText(objMap, "No detections in region", (x, y), 1, 1, (255))

    if (
        highest != None
        and highest[0] == x
        and highest[1] == y
        and lastCollectedX != x
        and lastCollectedY != y
    ):
        lastCollectedX = x
        lastCollectedY = y
        objectsCollected += 1
    cv2.putText(objMap, f"Collected: {objectsCollected}", (10, 50), 1, 2, (255))
    cv2.imshow(fieldMap.gameObjWindowName, objMap)
    return highest


def getMove(cx, cy, goalX, goalY, maxDistance) -> tuple[int, int]:
    dx = goalX - cx
    dy = goalY - cy
    mag = math.sqrt(dx**2 + dy**2)
    if mag > maxDistance:
        # rescale vectors as you are too far away to go in one step
        # unitize vectors
        dx /= mag
        dy /= mag
        # scale by max distance
        dx *= maxDistance
        dy *= maxDistance
    return (int(dx), int(dy))


def myRandom(random, a, b):
    return a + random * (b - a)


# for when there is no current target
def getRandomMove(robotX, robotY, fieldX, fieldY, maxDistance) -> tuple[int, int]:
    randDist = random.randint(int(maxDistance / 2), maxDistance)
    randAng = myRandom(random.random(), 0, 2 * math.pi)
    randDx = math.cos(randAng) * randDist
    randDy = math.sin(randAng) * randDist
    # handle clipping
    safetyoffset = 2
    if robotX + randDx > fieldX:
        randDx = fieldX - robotX - safetyoffset
    if robotX + randDx < 0:
        randDx = robotX + safetyoffset
    if robotY + randDy > fieldY:
        randDy = fieldY - robotY - safetyoffset
    if robotX + randDy < 0:
        randDy = robotY + safetyoffset

    return (int(randDx), int(randDy))


def startDemo() -> None:
    # default starting position for robot is half the field
    robotX = int(fieldX / 2)
    robotY = int(fieldY / 2)
    while True:
        Move = None
        highestCoords = getRangeHighest(robotX, robotY)
        if highestCoords is not None:
            print("Sucess!")
            (goalX, goalY) = highestCoords
            Move = getMove(
                robotX, robotY, goalX, goalY, maxRobotSpeed * 1
            )  # 1s for now
        else:
            # Move = getRandomMove(robotX, robotY, fieldX, fieldY, maxRobotSpeed * 1)
            Move = (0, 0)

        (moveX, moveY) = Move
        robotX += moveX
        robotY += moveY

        test_randomization_ranges(fieldMap, fieldMap.width, fieldMap.height)

        # slowly dissipate
        if random.randrange(0, 10) == 5:
            fieldMap.disspateOverTime(1)  # 1s

        k = cv2.waitKey(100) & 0xFF
        if k == ord("q"):
            return
        if k == ord("c"):
            fieldMap.clear_maps()


def test_randomization_ranges(map: ProbMap, width, height) -> None:
    for _ in range(2):
        x = random.randrange(0, width)
        y = random.randrange(0, height)
        # obj_size = 36*6 #size*total potential STD #random.randrange(36, 36)
        confidence = (
            random.randrange(65, 95, 1) / 100
        )  # generates a confidence threshold between 0.65 - 0.95
        try:
            map.addCustomObjectDetection(x, y, 100, 100, confidence)

        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    startDemo()
