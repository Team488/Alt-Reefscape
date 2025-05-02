
import cv2
from Captures import D435Capture, OAKCapture
from tools.Constants import D435IResolution, OAKDLITEResolution

if __name__ == "__main__":
    # pick your type
    # cap = D435Capture(res=D435IResolution.RS720P)
    cap = OAKCapture(res=OAKDLITEResolution.OAK1080P)

    while cap.isOpen():
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        color, depth = cap.getDepthAndColorFrame()

        cv2.imshow("depth", depth)
        cv2.imshow("color", color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break