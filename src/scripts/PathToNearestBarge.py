from JXTABLES.XTablesByteUtils import XTablesByteUtils
from JXTABLES.XTablesClient import XTablesClient

from pathplanning.nmc import fastMarchingMethodRPC
from JXTABLES import XTableValues_pb2 as XTableValues, XTableValues_pb2

midFieldX = 8.75665
robotWidthXInches = 35
robotCenterXInches = robotWidthXInches / 2
robotCenterXMeters = robotCenterXInches * 0.0254
blueMin = 4.20
redMin = 3.85
blueRobotMin = blueMin + robotCenterXMeters
redRobotMin = redMin - robotCenterXMeters
distanceFromBarge = 1

pose = None
team = None


def get_nearest_point(start_x, start_y, alliance):
    # If start_y falls between the red and blue min values, snap it
    # to the closest boundary.
    if alliance == XTableValues.Alliance.BLUE:
        if start_y <= blueRobotMin:
            start_y = blueRobotMin
    elif start_y >= redRobotMin:
        start_y = redRobotMin

    # Compute the X coordinate based on the alliance.

    if start_x < midFieldX:
        x = midFieldX - distanceFromBarge
        rot = 180
    else:
        x = midFieldX + distanceFromBarge
        rot = 0

    # Return the computed coordinates and the heading based on alliance.
    return x, start_y, rot


def run_periodic(xtables_client):
    if pose is None:
        return
    start_x = pose[0]
    start_y = pose[1]
    alliance = team
    if alliance is None:
        return
    xtableAlliance = (
        XTableValues.Alliance.BLUE if alliance == "Blue" else XTableValues.Alliance.RED
    )
    startPoint = XTableValues.ControlPoint(x=start_x, y=start_y)
    end = get_nearest_point(start_x, start_y, xtableAlliance)
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
        xtables_client.putBezierCurves("BEZIER_PATH_TO_NEAREST_BARGE", path)
    except Exception as e:
        print(e)


def update_pose(update):
    global pose
    pose2d = XTablesByteUtils.unpack_pose2d(update.value)
    if pose2d is None:
        return
    pose = pose2d


def update_team(update):
    global team
    value = XTablesByteUtils.to_string(update.value)
    if value is None:
        return
    team = value


def start():
    xtablesClient = XTablesClient(debug_mode=True, ip="localhost")
    xtablesClient.subscribe("PoseSubsystem.RobotPose", update_pose)
    xtablesClient.subscribe("TEAM", update_team)

    while True:
        try:
            run_periodic(xtablesClient)
        except Exception as e:
            print(e)
