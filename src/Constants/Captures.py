from Alt.Cameras.Captures import OpenCVCapture, CaptureWIntrinsics
from Alt.Cameras.Captures.OpenCVCapture import DefaultUseV4L2
from Alt.Cameras.Parameters.CameraIntrinsics import CameraIntrinsics


class OpenCvWIntrinsics(OpenCVCapture, CaptureWIntrinsics):
    def __init__(self, name : str, capturePath : str, cameraIntrinsics : CameraIntrinsics, useV4L2Backend : bool = DefaultUseV4L2, flushTimeMS : int = -1):
        super().__init__(name, capturePath, useV4L2Backend, flushTimeMS)
        self.setIntrinsics(cameraIntrinsics)