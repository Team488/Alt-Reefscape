if __name__ == "__main":
    path = "C:\\Users\\ademf\\OneDrive\\Documents\\MVI_7296.MP4"
    from tools.calibration import charuco_calibration_videos
    charuco_calibration_videos("outcalib.json",path)