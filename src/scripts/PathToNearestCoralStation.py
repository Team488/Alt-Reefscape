import math

from JXTABLES import XTableValues_pb2 as XTableValues, XTableValues_pb2
from JXTABLES.XTablesClient import XTablesClient, XTablesByteUtils

from pathplanning.nmc import fastMarchingMethodRPC

blueLeftCoralStation = (1.12, 7.03, -54.00)
blueRightCoralStation = (1.12, 1.03, 54.00)

redLeftCoralStation = (16.43, 1.03, 126.00)
redRightCoralStation = (16.43, 7.03, -126.00)

redCoralStations = [redRightCoralStation, redLeftCoralStation]
blueCoralStations = [blueRightCoralStation, blueLeftCoralStation]

pose = None
team = None


def get_nearest_point(start_x, start_y, points):
    return min(points, key=lambda p: math.dist((start_x, start_y), (p[0], p[1])))


def run_periodic(xtables_client):
    if pose is None:
        return
    start_x = pose[0]
    start_y = pose[1]
    if team is None:
        return
    xtableAlliance = (
        XTableValues.Alliance.BLUE if team == "Blue" else XTableValues.Alliance.RED
    )
    startPoint = XTableValues.ControlPoint(x=start_x, y=start_y)
    end = get_nearest_point(
        start_x,
        start_y,
        blueCoralStations
        if xtableAlliance == XTableValues.Alliance.BLUE
        else redCoralStations,
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
        xtables_client.putBezierCurves("BEZIER_PATH_TO_NEAREST_CORAL_STATION", path)
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
