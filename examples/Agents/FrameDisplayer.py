import time
from typing import Dict, Set, Any, Optional, Callable

import cv2
import numpy as np
from Core.Agents.Abstract.CameraUsingAgentBase import CameraUsingAgentBase
from tools.Constants import CameraIdOffsets2024
from coreinterface.FramePacket import FramePacket
from abstract.Agent import Agent


class FrameDisplayer(Agent):
    """Agent -> FrameDisplayer

    Agent that will automatically ingest frames and display them\n
    NOTE: Due to openCVs nature this agent must be run in the main thread\n
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.getSendEnableTable: Callable[[str], str] = lambda key: ""
        self.updateMap: Dict[str, np.ndarray] = {}
        self.keys: Set[str] = set()
        self.runFlag: bool = True
        self.displayedFrames = None

    def create(self) -> None:
        super().create()
        # perform agent init here (eg open camera or whatnot)
        if self.propertyOperator is None:
            raise ValueError("PropertyOperator not initialized")

        self.getSendEnableTable = (
            lambda key: f"{key}.{CameraUsingAgentBase.FRAMETOGGLEPOSTFIX}"
        )
        self.updateMap = {}
        self.keys = set()

        self.runFlag = True  # will be used with cv2 waitkey
        self.displayedFrames = self.propertyOperator.createReadOnlyProperty(
            "Showed_Frames", False
        )

    def subscribeFrameUpdate(self) -> None:
        if self.updateOp is None:
            raise ValueError("UpdateOperator not initialized")

        self.updateOp.subscribeAllGlobalUpdates(
            CameraUsingAgentBase.FRAMEPOSTFIX,
            updateSubscriber=self.__handleUpdate,
            runOnNewSubscribe=self.addKey,
            runOnRemoveSubscribe=self.removeKey,
        )

    def __handleUpdate(self, ret: Any) -> None:
        val = ret.value

        if not val or val == b"":
            return

        frame_pkt = FramePacket.fromBytes(val)
        frame = FramePacket.getFrame(frame_pkt)
        self.updateMap[ret.key] = frame

    def __showFrames(self) -> None:
        if self.displayedFrames is None:
            return

        showedFrames = False
        if len(self.updateMap.keys()) > 0:
            copy = dict(self.updateMap.items())  # copy
            for key, item in copy.items():
                frame = item
                if frame is not None:
                    cv2.imshow(key, frame)
                    showedFrames = True

        self.displayedFrames.set(showedFrames)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.runFlag = False

    def addKey(self, key: str) -> None:
        if self.xclient is None:
            raise ValueError("XTablesClient not initialized")

        cut = key[: key.rfind(".")]
        full = f"{cut}.{CameraUsingAgentBase.FRAMETOGGLEPOSTFIX}"
        self.xclient.putBoolean(full, True)
        print(f"{key=} added")
        print(f"{full=} set to True")
        # cv2.namedWindow(key)

    def removeKey(self, key: str) -> None:
        if self.xclient is None:
            raise ValueError("XTablesClient not initialized")

        cut = key[: key.rfind(".")]
        full = f"{cut}.{CameraUsingAgentBase.FRAMETOGGLEPOSTFIX}"
        self.xclient.putBoolean(full, False)
        print(f"{key=} removed")
        print(f"{full=} set to False")
        try:
            cv2.destroyWindow(key)
            cv2.waitKey(1)
        except Exception as e:
            print(e)
            pass

    def runPeriodic(self) -> None:
        super().runPeriodic()
        self.subscribeFrameUpdate()
        self.__showFrames()

    def onClose(self) -> None:
        super().onClose()
        if self.updateOp is not None:
            self.updateOp.unsubscribeToAllGlobalUpdates(
                CameraUsingAgentBase.FRAMEPOSTFIX, self.__handleUpdate
            )
        cv2.destroyAllWindows()

    def isRunning(self) -> bool:
        return self.runFlag

    def getDescription(self) -> str:
        return "Ingest_Frames_Show_Them"
