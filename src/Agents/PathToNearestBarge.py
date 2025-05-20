from JXTABLES import XTableValues_pb2 as XTableValues
from Alt.Core.Agents import Agent
from Alt.Pathplanning.nmc import fastMarchingMethodRPC


class PathToNearestBarge(Agent):
    midFieldX = 8.75665
    robotWidthXInches = 35
    robotCenterXInches = robotWidthXInches / 2
    robotCenterXMeters = robotCenterXInches * 0.0254
    blueMin = 4.20
    redMin = 3.85
    blueRobotMin = blueMin + robotCenterXMeters
    redRobotMin = redMin - robotCenterXMeters

    def create(self) -> None:
        self.bezierPathToNearestBarge = (
            self.propertyOperator.createCustomReadOnlyProperty(
                propertyTable="BEZIER_PATH_TO_NEAREST_BARGE",
                addBasePrefix=False,
                addOperatorPrefix=False,
            )
        )
        self.pose = self.propertyOperator.createCustomProperty(
            propertyTable="PoseSubsystem.RobotPose",
            propertyDefault=None,
            addBasePrefix=False,
            addOperatorPrefix=False,
            setDefaultOnNetwork=False,
        )
        self.distanceFromBarge = self.propertyOperator.createProperty(
            propertyTable="PathToNearestBarge.distanceFromBarge-m",
            propertyDefault=1,
        )
        self.team = self.propertyOperator.createCustomProperty(
            propertyTable="TEAM",
            propertyDefault=None,
            addBasePrefix=False,
            addOperatorPrefix=False,
            setDefaultOnNetwork=False,
        )

    def get_nearest_point(self, start_x, start_y, alliance):

        # If start_y falls between the red and blue min values, snap it
        # to the closest boundary.
        if alliance == XTableValues.Alliance.BLUE:
            if start_y <= self.blueRobotMin:
                start_y = self.blueRobotMin
        elif start_y >= self.redRobotMin:
            start_y = self.redRobotMin

        # Compute the X coordinate based on the alliance.

        if start_x < self.midFieldX:
            x = self.midFieldX - self.distanceFromBarge.get()
            rot = 180
        else:
            x = self.midFieldX + self.distanceFromBarge.get()
            rot = 0

        # Return the computed coordinates and the heading based on alliance.
        return x, start_y, rot

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
        end = self.get_nearest_point(start_x, start_y, xtableAlliance)
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
            path = fastMarchingMethodRPC.pathplan(request, inflation_max=90)
            if path is None:
                return
            self.bezierPathToNearestBarge.set(path)
        except Exception as e:
            print(e)

    def getIntervalMs(self):
        return -1

    def isRunning(self):
        return True

    def getDescription(self) -> str:
        return "Constant-Path-Generation-To-Nearest-Barge"
