import math
import cv2
import random

import numpy as np
from mapinternals.probmap import ProbMap
from tools import UnitConversion
from tools.Constants import CameraExtrinsics, CameraIntrinsics


def __myRandom(random, a, b):
    return a + random * (b - a)


lastRandAng = None


def getRandomMove(
    robotX, robotY, fieldX, fieldY, maxDistancePerMove, safetyOffset=2
) -> tuple[int, int]:
    randDist = random.randint(int(maxDistancePerMove / 2), maxDistancePerMove)
    randAng = __myRandom(random.random(), 0, 2 * math.pi)
    global lastRandAng
    if lastRandAng != None:
        influenceFactor = __myRandom(random.random(), 0.5, 0.7)
        randAng = randAng * influenceFactor + lastRandAng * (1 - influenceFactor)

    lastRandAng = randAng
    randDx = math.cos(randAng) * randDist
    randDy = math.sin(randAng) * randDist
    # handle clipping
    if robotX + randDx > fieldX:
        randDx = fieldX - robotX - safetyOffset
        lastRandAng = math.pi
    if robotX + randDx < 0:
        lastRandAng = 0
        randDx = robotX + safetyOffset
    if robotY + randDy > fieldY:
        lastRandAng = -math.pi / 2
        randDy = fieldY - robotY - safetyOffset
    if robotY + randDy < 0:
        lastRandAng = math.pi / 2
        randDy = robotY + safetyOffset

    return (int(randDx), int(randDy))


def getRealisticMoveVector(
    robotLocation, nextWaypoint, maxDistancePerMove
) -> tuple[int, int]:
    robotX, robotY = robotLocation
    nextWaypointX, nextWayPointY = nextWaypoint
    mvX, mvY = (nextWaypointX - robotX, nextWayPointY - robotY)
    dist = math.sqrt(mvX**2 + mvY**2)
    if dist <= maxDistancePerMove:
        return (mvX, mvY)
    else:
        return (mvX * maxDistancePerMove / dist, mvY * maxDistancePerMove / dist)


def rescaleCoordsDown(x, y, probmap: ProbMap):
    return x // probmap.resolution, y // probmap.resolution


def rescaleCoordsTogetherDown(coords, probmap: ProbMap):
    return coords[0] // probmap.resolution, coords[1] // probmap.resolution


def rescaleCoordsUp(x, y, probmap: ProbMap):
    return x * probmap.resolution, y * probmap.resolution


def rescaleCoordsTogetherUp(coords, probmap: ProbMap):
    return coords[0] * probmap.resolution, coords[1] * probmap.resolution


def drawRobotWithCams(
    frame,
    width,
    height,
    posX,
    posY,
    rotationRad,
    cams: list[tuple[CameraExtrinsics, CameraIntrinsics]],
    cameraLineLength=300,
) -> None:  # fov 90 deg  | fovLen = 70cm # camera is facing 45 to the left
    # drawing robot
    FrameOffset = math.atan((height / 2) / (width / 2))
    RobotAngLeft = rotationRad - FrameOffset
    RobotAngRight = rotationRad + FrameOffset
    FLx = int(posX + math.cos(RobotAngLeft) * width)
    FLy = int(posY + math.sin(RobotAngLeft) * height)
    FRx = int(posX + math.cos(RobotAngRight) * width)
    FRy = int(posY + math.sin(RobotAngRight) * height)

    BLx = int(posX - math.cos(RobotAngRight) * width)
    BLy = int(posY - math.sin(RobotAngRight) * height)
    BRx = int(posX - math.cos(RobotAngLeft) * width)
    BRy = int(posY - math.sin(RobotAngLeft) * height)
    cv2.line(frame, (FLx, FLy), (FRx, FRy), (0, 0, 255), 2)
    cv2.line(frame, (BLx, BLy), (BRx, BRy), (255, 0, 0), 2)
    cv2.line(frame, (BLx, BLy), (FLx, FLy), (255, 255, 255), 2)
    cv2.line(frame, (BRx, BRy), (FRx, FRy), (255, 255, 255), 2)

    for cam in cams:
        extr, intr = cam
        camOffsetX = extr.getOffsetXCM()
        camOffsetY = extr.getOffsetYCM()
        camYawRad = extr.getYawOffsetAsRadians()
        camFovRad = intr.getHFovRad()
        camX = (
            posX
            + camOffsetX * math.cos(rotationRad)
            - camOffsetY * math.sin(rotationRad)
        )
        camY = (
            posY
            + camOffsetX * math.sin(rotationRad)
            + camOffsetY * math.cos(rotationRad)
        )
        # drawing fov (from center of robot for now)
        cameraOffset = camYawRad
        rotLeft = (rotationRad + cameraOffset) - camFovRad / 2
        rotRight = (rotationRad + cameraOffset) + camFovRad / 2

        LeftX = int(camX + math.cos(rotLeft) * cameraLineLength)
        LeftY = int(camY + math.sin(rotLeft) * cameraLineLength)

        RightX = int(camX + math.cos(rotRight) * cameraLineLength)
        RightY = int(camY + math.sin(rotRight) * cameraLineLength)

        camX = int(camX)
        camY = int(camY)

        cv2.line(frame, (camX, camY), (LeftX, LeftY), (255, 130, 0), 1)
        cv2.line(frame, (camX, camY), (RightX, RightY), (255, 130, 0), 1)


def drawRobotWithCam(
    frame,
    width,
    height,
    posX,
    posY,
    rotationRad,
    camOffsetX,
    camOffsetY,
    camYawRad,
    camFovRad,
    cameraLineLength=300,
) -> None:  # fov 90 deg  | fovLen = 70cm # camera is facing 45 to the left
    # drawing robot
    FrameOffset = math.atan((height / 2) / (width / 2))
    RobotAngLeft = rotationRad - FrameOffset
    RobotAngRight = rotationRad + FrameOffset
    FLx = int(posX + math.cos(RobotAngLeft) * width)
    FLy = int(posY + math.sin(RobotAngLeft) * height)
    FRx = int(posX + math.cos(RobotAngRight) * width)
    FRy = int(posY + math.sin(RobotAngRight) * height)

    BLx = int(posX - math.cos(RobotAngRight) * width)
    BLy = int(posY - math.sin(RobotAngRight) * height)
    BRx = int(posX - math.cos(RobotAngLeft) * width)
    BRy = int(posY - math.sin(RobotAngLeft) * height)
    cv2.line(frame, (FLx, FLy), (FRx, FRy), (0, 0, 255), 2)
    cv2.line(frame, (BLx, BLy), (BRx, BRy), (255, 0, 0), 2)
    cv2.line(frame, (BLx, BLy), (FLx, FLy), (255, 255, 255), 2)
    cv2.line(frame, (BRx, BRy), (FRx, FRy), (255, 255, 255), 2)

    camX = (
        posX + camOffsetX * math.cos(rotationRad) - camOffsetY * math.sin(rotationRad)
    )
    camY = (
        posY + camOffsetX * math.sin(rotationRad) + camOffsetY * math.cos(rotationRad)
    )
    # drawing fov (from center of robot for now)
    cameraOffset = camYawRad
    rotLeft = (rotationRad + cameraOffset) - camFovRad / 2
    rotRight = (rotationRad + cameraOffset) + camFovRad / 2

    LeftX = int(camX + math.cos(rotLeft) * cameraLineLength)
    LeftY = int(camY + math.sin(rotLeft) * cameraLineLength)

    RightX = int(camX + math.cos(rotRight) * cameraLineLength)
    RightY = int(camY + math.sin(rotRight) * cameraLineLength)

    camX = int(camX)
    camY = int(camY)

    cv2.line(frame, (camX, camY), (LeftX, LeftY), (255, 130, 0), 1)
    cv2.line(frame, (camX, camY), (RightX, RightY), (255, 130, 0), 1)


def drawRobot(
    frame,
    width,
    height,
    posX,
    posY,
    rotationRad,
) -> None:  # fov 90 deg  | fovLen = 70cm # camera is facing 45 to the left
    # drawing robot
    FrameOffset = math.atan((height / 2) / (width / 2))
    RobotAngLeft = rotationRad - FrameOffset
    RobotAngRight = rotationRad + FrameOffset
    FLx = int(posX + math.cos(RobotAngLeft) * width)
    FLy = int(posY + math.sin(RobotAngLeft) * height)
    FRx = int(posX + math.cos(RobotAngRight) * width)
    FRy = int(posY + math.sin(RobotAngRight) * height)

    BLx = int(posX - math.cos(RobotAngRight) * width)
    BLy = int(posY - math.sin(RobotAngRight) * height)
    BRx = int(posX - math.cos(RobotAngLeft) * width)
    BRy = int(posY - math.sin(RobotAngLeft) * height)
    cv2.line(frame, (FLx, FLy), (FRx, FRy), (0, 0, 255), 2)
    cv2.line(frame, (BLx, BLy), (BRx, BRy), (255, 0, 0), 2)
    cv2.line(frame, (BLx, BLy), (FLx, FLy), (255, 255, 255), 2)
    cv2.line(frame, (BRx, BRy), (FRx, FRy), (255, 255, 255), 2)


def drawBox(frame, bboxXYXY, class_str, conf, color=(10, 100, 255), buffer=8):
    p1 = np.array(UnitConversion.toint(bboxXYXY[:2]))
    p2 = np.array(UnitConversion.toint(bboxXYXY[2:]))
    text = f"{class_str} Conf:{conf:.2f}"
    cv2.rectangle(frame, p1, p2, color, 3, 0)

    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 3
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    textStart = UnitConversion.toint(p1 - np.array([0, text_height + buffer]))
    textEnd = UnitConversion.toint(p1 + np.array([text_width + buffer, 0]))
    cv2.rectangle(frame, textStart, textEnd, color, -1)

    cv2.putText(
        frame,
        text,
        p1 + np.array([int(buffer / 2), -int(buffer / 2)]),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
