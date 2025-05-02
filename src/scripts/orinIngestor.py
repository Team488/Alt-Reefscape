import time

from Captures import FileCapture
import cv2
from time import strftime, localtime


def getTimeStr():
    return strftime("%Y-%m-%d_%H-%M-%S", localtime())


def startWriting():

    port = 1181
    streamName = "stream.mjpg"
    hostnames = [
        "photonvisionfrontright.local",
        "photonvisionfrontleft.local",
        "photonvisionback.local",
    ]

    captures = [
        FileCapture(f"http://{hostname}:{port}/{streamName}") for hostname in hostnames
    ]

    videowriters = []
    for capture, hostname in zip(captures, hostnames):
        capture.create()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        filePath = f"{hostname}_{getTimeStr()}.mp4"
        writer = cv2.VideoWriter(
            filePath,
            fourcc,
            capture.getFps(),
            capture.getFrameShape()[:2][::-1],
        )
        videowriters.append(writer)

    while captures:
        to_remove = []

        for capture, writer in zip(captures, videowriters):
            if not capture.isOpen():
                capture.close()
                writer.release()
                to_remove.append((capture, writer))
                continue  # Move to next capture

            frame = capture.getMainFrame()
            if frame is None:
                continue  # Skip writing if no frame is available

            writer.write(frame)  # Write frame only if valid

        # Remove closed captures after iteration
        for capture, writer in to_remove:
            captures.remove(capture)
            videowriters.remove(writer)
        time.sleep(0.01)

    for capture, writer in zip(captures, videowriters):
        capture.close()
        writer.release()
    # Execution reaches here once all captures close
    print("All video captures have closed.")


if __name__ == "__main__":
    startWriting()
