from mapinternals.probmap import ProbMap
from tools.Constants import Label
import cv2


def startDemo() -> None:

    labels = [Label.NOTE, Label.ROBOT, Label.ALGAE]
    isMouseDown = [False for _ in labels]

    res = 5  # cm
    # object and robot values not necessary here
    map = ProbMap(labels, 2000, 1000, res)

    def mouseDownCallback(event, label_idx, x, y, flags, param) -> None:
        nonlocal isMouseDown
        if event == cv2.EVENT_LBUTTONDOWN:
            isMouseDown[label_idx] = True
            #  print("clicked at ", x," ", y)
            map.addCustomObjectDetection(label_idx, x, y, 15, 14, 0.9)
        elif event == cv2.EVENT_MOUSEMOVE:
            if isMouseDown[label_idx]:
                #   print("dragged at ", x," ", y)
                map.addCustomObjectDetection(label_idx, x, y, 15, 15, 0.9)
        elif event == cv2.EVENT_LBUTTONUP:
            isMouseDown[label_idx] = False

    for idx, label in enumerate(labels):
        wname = str(label)
        cv2.namedWindow(wname)
        cv2.setMouseCallback(
            wname,
            lambda event, x, y, flags, param, idx=idx: mouseDownCallback(
                event, idx, x, y, flags, param
            ),
        )

    while True:
        map.disspateOverTime(1)  # 1s
        map.displayHeatMaps()
        for idx, label in enumerate(labels):
            print(f"Best {str(label)}:", map.getHighestObject(idx))

        k = cv2.waitKey(100) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            map.clear_maps()


if __name__ == "__main__":
    startDemo()
