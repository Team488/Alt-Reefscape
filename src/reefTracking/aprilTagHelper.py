import json

import numpy as np
from Alt.Cameras.Parameters.CameraIntrinsics import CameraIntrinsics
from robotpy_apriltag import (
    AprilTagField,
    AprilTagFieldLayout,
    AprilTagDetector,
    AprilTagPoseEstimator,
    AprilTagPoseEstimate,
)


class AprilTagLocal:
    def __init__(self, cameraIntrinsic: CameraIntrinsics) -> None:
        self.detector = AprilTagDetector()
        self.detectorConfig = AprilTagDetector.Config()
        self.detectorConfig.quadDecimate = 1
        self.detectorConfig.quadSigma = 0.5
        self.detector.setConfig(self.detectorConfig)
        self.detector.addFamily("tag36h11")

        self.fx = cameraIntrinsic.getFx()
        self.fy = cameraIntrinsic.getFy()
        self.cx = cameraIntrinsic.getCx()
        self.cy = cameraIntrinsic.getCy()

        # Tag Size: 165.1 mm = 0.1651 m
        config = AprilTagPoseEstimator.Config(
            tagSize=0.1651, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
        )
        print(
            "camera intrinsics: {cx, cy, fx, fy}:", self.cx, self.cy, self.fx, self.fy
        )

        self.estimator = AprilTagPoseEstimator(config)

    def loadConfig(self, config_file) -> None:
        try:
            with open(config_file) as PV_config:
                data = json.load(PV_config)

                self.cameraIntrinsics = data["cameraIntrinsics"]["data"]
                self.fx = self.cameraIntrinsics[0]
                self.fy = self.cameraIntrinsics[4]
                self.cx = self.cameraIntrinsics[2]
                self.cy = self.cameraIntrinsics[5]

                self.K = np.array(
                    [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
                    dtype=np.float32,
                )

                self.width = int(data["resolution"]["width"])
                self.height = int(data["resolution"]["height"])

                distCoeffsSize = int(data["distCoeffs"]["cols"])
                self.distCoeffs = np.array(
                    data["distCoeffs"]["data"][0:distCoeffsSize], dtype=np.float32
                )
        except Exception as e:
            print(f"Failed to open config! {e}")

    def getDetections(self, grayscale_image):
        output = self.detector.detect(grayscale_image)
        return output

    def getOrthogonalEstimates(self, output) -> list[AprilTagPoseEstimate]:
        estimates = []
        for detections in output:
            # Retrieve the corners of the AT detection
            # Get Tag Pose Information
            # tag_pose_estimation = AprilTagPoseEstimator.estimate(estimator, detections)
            tag_pose_estimation_orthogonal = (
                AprilTagPoseEstimator.estimateOrthogonalIteration(
                    self.estimator, detections, 500
                )
            )
            estimates.append(tag_pose_estimation_orthogonal)
        return estimates
