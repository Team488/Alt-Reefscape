from enum import Enum

class CommonVideos(Enum):
    ReefscapeCompilation = "assets/reefscapevid.mp4"
    Comp2024Clip = "assets/video12qual25clipped.mp4"
    ArucoCalib = "assets/arucoCalib.mp4"
    StingerCam = "assets/StingerCam.mp4"

    @property
    def path(self):
        return self.value


class SimulationEndpoints(Enum):
    FRONTRIGHTSIM = "http://localhost:3000/Robot_FrontRight%20Camera?dummy=param.mjpg"
    FRONTRIGHTAPRILTAGSIM = (
        "http://localhost:3000/Robot_FrontRight%20CameraAT?dummy=param.mjpg"
    )
    FRONTLEFTSIM = "http://localhost:3000/Robot_FrontLeft%20Camera?dummy=param.mjpg"
    REARRIGHTSIM = "http://localhost:3000/Robot_RearRight%20Camera?dummy=param.mjpg"
    REARLEFTSIM = "http://localhost:3000/Robot_RearLeft%20Camera?dummy=param.mjpg"

    @property
    def path(self):
        return self.value



class RealSenseSerialIDS(Enum):
    FRONTLEFTDEPTHSERIALID = "048522074864"
    FRONTRIGHTDEPTHSERIALID = "843112072752"