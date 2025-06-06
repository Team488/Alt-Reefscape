import cv2

from Alt.Core.Agents import Agent
from Alt.Core.Utils.timeFmt import getTimeStr
from Alt.Cameras.Captures import OpenCVCapture


class orinIngestorAgent(Agent):
    port = 1181
    streamName = "stream.mjpg"
    hostnames = [
        "photonvisionfrontright.local",
        "photonvisionfrontleft.local",
        "photonvisionback.local",
    ]

    def create(self) -> None:
        # for example here i can create a propery to configure what to call myself
        self.captures = [
            OpenCVCapture("orinFeed", f"http://{hostname}:{self.port}/{self.streamName}")
            for hostname in self.hostnames
        ]

        self.videowriters = []
        for capture, hostname in zip(self.captures, self.hostnames):
            capture.create()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            filePath = f"{hostname}_{getTimeStr()}.mp4"
            writer = cv2.VideoWriter(
                filePath,
                fourcc,
                capture.getFps(),
                capture.getFrameShape()[:2][::-1],
            )
            self.videowriters.append(writer)

    def runPeriodic(self) -> None:
        to_remove = []

        for capture, writer in zip(self.captures, self.videowriters):
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
            self.captures.remove(capture)
            self.videowriters.remove(writer)

    def onClose(self) -> None:
        pass

    def isRunning(self) -> bool:
        return len(self.captures) > 0

    def forceShutdown(self) -> None:
        for cap in self.captures:
            cap.close()

        for writer in self.videowriters:
            writer.release()

    def getDescription(self) -> str:
        return "Ingest-Photonvision-Streams-Write-To-File"

    def getIntervalMs(self) -> int:
        return -1
