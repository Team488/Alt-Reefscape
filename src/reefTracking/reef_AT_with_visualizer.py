"""
https://robotpy.readthedocs.io/projects/apriltag/en/latest/robotpy_apriltag.html
https://pypi.org/project/photonlibpy/
PORT TO PHOTONVISION LATER: https://docs.photonvision.org/en/latest/docs/programming/photonlib/index.html

Steps:
1. Get the pose of AT
2. Find the 3D offsets of the AT -> reef branches (have a cad model for that)
3. Transform locations of reef branches via depth data -> camera frame
4. Detect if objects are "fixed" onto those branches

Measurement in inches
"""
import json
from threading import Thread
import cv2
import numpy as np
from robotpy_apriltag import (
    AprilTagDetector,
    AprilTagPoseEstimator,
)

from ultralytics import YOLO


from .Reef import Reef, Alliance
from .ReefVisualizer import ReefVisualizer

def run_AT_detection(app, image) -> None:
    global width, height
    image = cv2.undistort(image, K, distCoeffs)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # inferencing
    results = model.predict(image, show_boxes=True, conf=0.8, show=False, verbose=False)

    annotated_frame = results[0].plot()
    boxes = results[0].boxes.xywh.cpu()
    confs = results[0].boxes.conf.cpu()
    ids = results[0].boxes.cls.cpu()
    object_ct = 0
    output = detector.detect(grayscale_image)

    # Loops through known AT detections
    detection_ids = [detection.getId() for detection in output]  # ids of what we see
    known_tags = (
        reef.get_self_alliance_tags()
    )  # retrieves all reef tags for our particular alliance color
    reef_tags_detected = list(
        set(known_tags) & set(detection_ids)
    )  # finds intersecting tags (so we don't detect tags we don't care about)
    tags_not_in_view = list(
        set(known_tags) ^ set(detection_ids)
    )  # finds the non-intersecting tags to update visualizer

    for detections in output:
        detection_id = detections.getId()
        if detection_id not in reef_tags_detected:
            continue  # Skip IDs that we do not care about

        print("ID", detection_id)

        # Retrieve the corners of the AT detection
        points = []
        for corner in range(0, 4):
            x = detections.getCorner(corner).x
            y = detections.getCorner(corner).y
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 255), thickness=3)

        # Retrieve the center of the AT detection
        centerX = detections.getCenter().x
        centerY = detections.getCenter().y

        cv2.circle(
            image, (int(centerX), int(centerY)), 2, color=(0, 255, 255), thickness=3
        )
        # Get Tag Pose Information
        tag_pose_estimation_orthogonal = (
            AprilTagPoseEstimator.estimateOrthogonalIteration(
                estimator, detections, 500
            )
        )
        tag_pose_estimation_orthogonal_pose1_matrix = (
            tag_pose_estimation_orthogonal.pose1
        )
        print(
            f"orthogonal pose 1: x: {tag_pose_estimation_orthogonal_pose1_matrix.x}, y: {tag_pose_estimation_orthogonal_pose1_matrix.y}, z: {tag_pose_estimation_orthogonal_pose1_matrix.z}"
        )

        for offset_idx, offset_3d in cad_to_branch_offset.items():
            # solve camera -> branch via camera -> tag and tag -> branch transformations
            tag_to_reef_homography = np.append(offset_3d, 1.0)  # ensures shape is 4x4
            # camera_to_reef = np.dot(tag_pose_estimation_matrix, tag_to_reef_homography)
            camera_to_reef = np.dot(
                tag_pose_estimation_orthogonal_pose1_matrix.toMatrix(),
                tag_to_reef_homography,
            )

            x_cam, y_cam, z_cam, _ = camera_to_reef

            # project the 3D point to 2D image coordinates:
            u = (fx * x_cam / z_cam) + cx
            v = (fy * y_cam / z_cam) + cy

            # Retrieve the branches we're looking at
            level_str = offset_idx[0:2]
            direction_str = offset_idx[3:4]
            level = Reef.Level[level_str]

            branches = reef.get_branches_at_tag(
                detection_id
            )  # This returns 17 => [A, B]
            branch_index = (
                0 if "L" in direction_str else 1
            )  # Left or Right Indexing 0 = A, 1 = B
            branch = branches[branch_index]  # Returns this Branch object from Reef.py

            branch_name = branch.value[1]  # Returns the Branch "string" value
            level_name = level.value[1]  # Returns the Level "string value"
            branch_level_index_str = f"{branch_name}_{level_name}"  # "A_L1"

            # Check if U & V is in the frame:
            if (0 < u < width) and (0 < v < height):
                # Adds the known poles of what is in frame
                cv2.circle(image, (int(u), int(v)), 5, (0, 255, 255), 2)
                cv2.putText(
                    image,
                    f"{offset_idx}",
                    (int(u), int(v) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                # Check if there are any detections, if not, we're assuming the box is clear
                if len(boxes) == 0:
                    print("No detections")
                    app.queue_color_update(branch_level_index_str, colors["green"])
                else:
                    # There are detections, we will now check if it's on the branch
                    print("BOXES IS NOT EMPTY")
                    for box in boxes:
                        x_center, y_center, width, height = box.tolist()

                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_center + width / 2
                        y_max = y_center + height / 2

                        # Is the coral within the bounding boxes?
                        if (x_min <= u <= x_max) and (y_min <= v <= y_max):
                            reef.set_branch_state(branch, level, Reef.CoralState.ON)
                            app.queue_color_update(
                                branch_level_index_str, colors["red"]
                            )  # RED = Filled
                        else:
                            app.queue_color_update(
                                branch_level_index_str, colors["green"]
                            )  # Green = Space is unoccupied
            else:
                # That particular branch level could not be seen. Mark as it as "unknown" yellow
                app.queue_color_update(branch_level_index_str, colors["yellow"])

    # UPDATE BRANCH STATES WE DONT SEE
    for tag in tags_not_in_view:
        branches = reef.get_branches_at_tag(tag)
        for branch in branches:
            for level in Reef.Level:
                branch_level_index_str = f"{branch.value[1]}_{level.value[1]}"  # .value retrieves the string counterpart
                # Make sure it's not occupied already (we're assuming no descores)
                if not (reef.get_branch_state_at(branch, level) == Reef.CoralState.ON):
                    app.queue_color_update(branch_level_index_str, colors["yellow"])

    reconvert_grayscale = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    stacked_frames = cv2.hconcat([image, annotated_frame])
    cv2.imshow("display", stacked_frames)
    print("================================")
    for branch, level in reef.get_branch_with_state(Reef.CoralState.ON):
        print("Filled", branch, level)


if "__main__" == __name__:

    detector = AprilTagDetector()
    detector.addFamily("tag36h11")

    colors = {"red": (1, 0, 0, 1), "green": (0, 1, 0, 1), "yellow": (1, 1, 0, 1)}

    ### CAD Offset:
    ### X = Downwards -> Right
    ### Y = Upwards -> Right
    ### Z = Upwards

    # Measurement is in Inches

    # CAD to tip of the rod. (MAX Distance)
    cad_to_branch_offset = {
        "L2-L": np.array([-6.756, -19.707, 2.608]),
        "L2-R": np.array([6.754, -19.707, 2.563]),
        "L3-L": np.array([-6.639, -35.606, 2.628]),
        "L3-R": np.array([6.637, -35.606, 2.583]),
        "L4-L": np.array([-6.470, -58.4175, 0.921]),  # NOT MODIFIED
        "L4-R": np.array([6.468, -58.4175, 0.876]),  # NOT MODIFIED
    }
    ### Camera AT Coordinate System:
    #   X is LEFT -> Right  [-inf, inf]
    #   Y is TOP -> Down    [-inf, inf]
    #   Z is DEPTH AWAY     [0, inf]

    # Convert to meters:
    for branch, offset in cad_to_branch_offset.items():
        for i in range(len(offset)):
            offset[i] *= 0.0254

    # Obtain camera calibration data
    with open(
        "/home/jlwu/Development/FRC/Reefscape/config/1280x800v1.json"
    ) as PV_config:
        data = json.load(PV_config)

        cameraIntrinsics = data["cameraIntrinsics"]["data"]
        fx = cameraIntrinsics[0]
        fy = cameraIntrinsics[4]
        cx = cameraIntrinsics[2]
        cy = cameraIntrinsics[5]

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        width = int(data["resolution"]["width"])
        height = int(data["resolution"]["height"])

        distCoeffsSize = int(data["distCoeffs"]["cols"])
        distCoeffs = np.array(
            data["distCoeffs"]["data"][0:distCoeffsSize], dtype=np.float32
        )

    # Start Capture and Calibrate Camera
    # video_path = "video/2.mkv" # or do int 0 for /dev/video0
    video_path = 4
    cap = cv2.VideoCapture(video_path)  # /dev/video0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("camera intrinsics: {cx, cy, fx, fy}:", cx, cy, fx, fy)

    # Tag Size: 165.1 mm = 0.1651 m
    config = AprilTagPoseEstimator.Config(tagSize=0.1651, fx=fx, fy=fy, cx=cx, cy=cy)
    estimator = AprilTagPoseEstimator(config)

    model = YOLO("/home/jlwu/Development/FRC/Reefscape/best-137.pt")

    # Set up your visualizer and build the app instance
    reef = Reef(Alliance.BLUE)
    app = ReefVisualizer()
    app_instance = app.build()

    # Define a function that continuously reads frames and runs detection
    def detection_loop(app_instance) -> None:
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            # Call detection with the correct app instance and image
            run_AT_detection(app_instance, image)

            # Allow for graceful exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        cap.release()  # Release the video capture when done

    # Start the detection loop in a separate thread
    backend_thread = Thread(target=detection_loop, args=(app_instance,))
    backend_thread.daemon = True
    backend_thread.start()

    # Run the visualizer GUI on the main thread
    app.run()
