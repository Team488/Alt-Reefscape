import math

import cv2
import numpy as np

from abstract.Agent import Agent
from pathplanning.nmc import fastMarchingMethodRPC
from JXTABLES import XTableValues_pb2 as XTableValues


class PathToNearestCoralStation(Agent):
    blueLeftCoralStation = (1.12, 7.03, -54.00)
    blueRightCoralStation = (1.12, 1.03, 54.00)

    redLeftCoralStation = (16.43, 1.03, 126.00)
    redRightCoralStation = (16.43, 7.03, -126.00)

    redCoralStations = [redRightCoralStation, redLeftCoralStation]
    blueCoralStations = [blueRightCoralStation, blueLeftCoralStation]

    def create(self) -> None:
        self.bezierPathToNearestCoralStation = (
            self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable="BEZIER_PATH_TO_NEAREST_CORAL_STATION",
                addBasePrefix=False,
                addOperatorPrefix=False,
            )
        )
        self.pose = self.propertyOperator.createProperty(
            propertyTable="PoseSubsystem.RobotPose",
            propertyDefault=None,
            addBasePrefix=False,
            addOperatorPrefix=False,
            setDefaultOnNetwork=False,
            isCustom=True,
        )
        self.team = self.propertyOperator.createProperty(
            propertyTable="TEAM",
            propertyDefault=None,
            addBasePrefix=False,
            addOperatorPrefix=False,
            setDefaultOnNetwork=False,
            isCustom=True,
        )

    def get_nearest_point(self, start_x, start_y, points):
        return min(points, key=lambda p: math.dist((start_x, start_y), (p[0], p[1])))

    def runPeriodic(self) -> None:
        start = self.pose.get()
        if start is None:
            return
        start_x = start[0]
        start_y = start[1]
        alliance = self.team.get()
        if alliance is None:
            return
        xtableAlliance = (
            XTableValues.Alliance.BLUE
            if alliance == "Blue"
            else XTableValues.Alliance.RED
        )
        startPoint = XTableValues.ControlPoint(x=start_x, y=start_y)
        end = self.get_nearest_point(
            start_x,
            start_y,
            self.blueCoralStations
            if xtableAlliance == XTableValues.Alliance.BLUE
            else self.redCoralStations,
        )
        endPoint = XTableValues.ControlPoint(x=end[0], y=end[1])
        arguments = XTableValues.AdditionalArguments(alliance=xtableAlliance)
        options = XTableValues.TraversalOptions(
            metersPerSecond=5,
            finalRotationDegrees=end[2],
            accelerationMetersPerSecond=5,
        )
        request = XTableValues.RequestVisionCoprocessorMessage(
            start=startPoint, end=endPoint, arguments=arguments, options=options
        )
        try:
            path = fastMarchingMethodRPC.pathplan(request)
            if path is None:
                return
            self.bezierPathToNearestCoralStation.set(path)
        except Exception as e:
            print(e)

    def getIntervalMs(self):
        return -1

    def isRunning(self):
        return True

    def getDescription(self) -> str:
        return "Constant-Path-Generation-To-Nearest-Coral-Station"
