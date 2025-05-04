import time
from Core import COREMODELTABLE, COREINFERENCEMODE
from Core.Central import Central
from tools.Constants import CameraIdOffsets2024, ATLocations, Units, TEAM
from coreinterface.DetectionPacket import DetectionPacket
from coreinterface.ReefPacket import ReefPacket
from abstract.Agent import Agent
from Core.Agents.Partials.ObjectLocalizingAgentBase import ObjectLocalizingAgentBase
from Core.Agents.Abstract.ReefTrackingAgentBase import ReefTrackingAgentBase
from Core.Agents.Abstract.PositionLocalizingAgentBase import PositionLocalizingAgentBase

from JXTABLES import XTableProto_pb2 as XTableProto
from JXTABLES import XTableValues_pb2 as XTableValue
from typing import List


class CentralAgent(PositionLocalizingAgentBase):
    """Agent -> CentralAgentBase

    Adds automatic ingestion of detection packets into the central process
    """

    def create(self) -> None:
        # put inference mode on xtables, so local observers can assert they are running the same model type
        self.xclient.putString(COREMODELTABLE, COREINFERENCEMODE.getName())

        super().create()
        self.central = Central()
        # perform agent init here (eg open camera or whatnot)
        print("CREATED HOST")
        self.objectupdateMap = {}
        self.localObjectUpdateMap = {}
        self.reefupdateMap = {}
        self.localReefUpdateMap = {}
        self.lastUpdateTimeMs = -1

        self.iterationsPerUpdate = 50
        self.iter_count = 0

        self.clAT = self.propertyOperator.createCustomReadOnlyProperty(
            "BESTOPENREEF_AT", None, addBasePrefix=False
        )
        self.clBR = self.propertyOperator.createCustomReadOnlyProperty(
            "BESTOPENREEFBRANCH", None, addBasePrefix=False
        )
        self.bestObjs = []
        for label in self.central.labels:
            boX = self.propertyOperator.createCustomReadOnlyProperty(
                f"Best.{str(label)}.x", -1, addBasePrefix=False
            )
            boY = self.propertyOperator.createCustomReadOnlyProperty(
                f"Best.{str(label)}.y", -1, addBasePrefix=False
            )
            boP = self.propertyOperator.createCustomReadOnlyProperty(
                f"Best.{str(label)}.prob", -1, addBasePrefix=False
            )

            nX = self.propertyOperator.createCustomReadOnlyProperty(
                f"Nearest.{str(label)}.x", -1, addBasePrefix=False
            )
            nY = self.propertyOperator.createCustomReadOnlyProperty(
                f"Nearest.{str(label)}.y", -1, addBasePrefix=False
            )
            nP = self.propertyOperator.createCustomReadOnlyProperty(
                f"Nearest.{str(label)}.prob", -1, addBasePrefix=False
            )
            self.bestObjs.append((boX, boY, boP, nX, nY, nP))

        self.reefmap_states = self.propertyOperator.createCustomReadOnlyProperty(
            "REEFMAP_STATES", None, addBasePrefix=False
        )

        self.reefmap_states_proto = self.propertyOperator.createReadOnlyProperty(
            "REEFMAP_STATES_PROTO",
            None,
        )

        self.offsetMap = {}
        self.offsetLength = 30

        self.team = self.getTeam()

    def addKeyObject(self, key):
        self.objectupdateMap[key] = ([], 0, 0)
        self.localObjectUpdateMap[key] = 0

    def addKeyReef(self, key):
        self.reefupdateMap[key] = ([], 0)
        self.localReefUpdateMap[key] = 0

    # handles a subscriber update from one of the cameras
    def __handleObjectUpdate(self, ret) -> None:
        val = ret.value
        key = ret.key
        if key in self.offsetMap:
            idOffset = self.offsetMap[key]
        else:
            idOffset = (len(self.offsetMap) + 1) * self.offsetLength
            self.offsetMap[key] = idOffset

        lastidx = self.objectupdateMap[key][2]
        lastidx += 1
        if not val or val == b"":
            return
        det_packet = DetectionPacket.fromBytes(val)
        # print(f"{det_packet.timestamp=}")
        packet = (DetectionPacket.toDetections(det_packet), idOffset, lastidx)
        self.objectupdateMap[key] = packet

    def __handleReefUpdate(self, ret) -> None:
        val = ret.value
        key = ret.key
        lastidx = self.reefupdateMap[key][1]
        lastidx += 1
        if not val or val == b"":
            return
        reef_packet = ReefPacket.fromBytes(val)
        # print(f"{det_packet.timestamp=}")
        packet = (ReefPacket.getFlattenedObservations(reef_packet), lastidx)
        self.reefupdateMap[key] = packet

    def __centralUpdate(self) -> None:
        currentTime = time.time() * 1000
        if self.lastUpdateTimeMs == -1:
            timePerLoopMS = 50  # random default value
        else:
            timePerLoopMS = currentTime - self.lastUpdateTimeMs
        self.lastUpdateTimeMs = currentTime

        accumulatedObjectResults = []
        accumulatedReefResults = []
        for key in self.localObjectUpdateMap.keys():
            # objects
            localidx = self.localObjectUpdateMap[key]
            resultpacket = self.objectupdateMap[key]
            res, packetidx = resultpacket[:2], resultpacket[2]
            if packetidx != localidx:
                # no update same id
                self.localObjectUpdateMap[key] = packetidx
                accumulatedObjectResults.append(res)

        # reef

        for key in self.localReefUpdateMap.keys():
            localidx = self.localReefUpdateMap[key]
            resultpacket = self.reefupdateMap[key]
            res, packetidx = resultpacket[0], resultpacket[1]
            # print("key=", key, "result=", res)
            if packetidx != localidx:
                # no update same id
                self.localReefUpdateMap[key] = packetidx
                accumulatedReefResults.append(res)

        # update objects
        self.central.processFrameUpdate(
            cameraResults=accumulatedObjectResults, timeStepMs=timePerLoopMS
        )
        # update reef
        # print("Updating Reef with accumulatedReefResults", accumulatedReefResults)
        self.central.processReefUpdate(
            reefResults=accumulatedReefResults, timeStepMs=timePerLoopMS
        )

    def periodicSubscribe(self):
        self.updateOp.subscribeAllGlobalUpdates(
            ReefTrackingAgentBase.OBSERVATIONPOSTFIX,
            self.__handleReefUpdate,
            runOnNewSubscribe=self.addKeyReef,
        )
        self.updateOp.subscribeAllGlobalUpdates(
            ObjectLocalizingAgentBase.DETECTIONPOSTFIX,
            self.__handleObjectUpdate,
            runOnNewSubscribe=self.addKeyObject,
        )

    def runPeriodic(self) -> None:
        super().runPeriodic()
        self.__centralUpdate()
        self.putBestNetworkValues()
        self.iter_count += 1
        if self.iter_count == self.iterationsPerUpdate:
            # reset the count
            self.iter_count = 0
            self.periodicSubscribe()

    def putBestNetworkValues(self) -> None:
        # Send the ReefPacket for the entire map
        import time

        timestamp = time.time()

        mapstate_packet = self.central.reefState.getReefMapState_as_ReefPacket(
            team=self.team, timestamp=timestamp
        )
        bytes = mapstate_packet.to_bytes()
        self.reefmap_states.set(bytes)

        # send protobuf
        probability_map_detections = self.getProbabilityMapDetectionsProtobuf()
        # print("probability map detections", probability_map_detections, type(probability_map_detections))
        self.reefmap_states_proto.set(probability_map_detections)

        # Send the confidence of highest algae
        for idx, (bX, bY, bP, nX, nY, nP) in enumerate(self.bestObjs):
            highest = self.central.objectmap.getHighestObject(class_idx=idx)
            bX.set(highest[0])
            bY.set(highest[1])
            bP.set(float(highest[2]))

            nearest = self.central.objectmap.getNearestAboveThreshold(
                idx, self.robotPose2dCMRAD[:2], threshold=0.3, team=None
            )  # update
            if nearest is not None:
                nX.set(float(nearest[0]))
                nY.set(float(nearest[1]))
                nP.set(float(nearest[3]))

        closest_At, closest_branch = self.central.reefState.getClosestOpen(
            self.robotPose2dCMRAD, threshold=0.4, team=self.getTeam()
        )
        if closest_At is None:
            closest_At = -1
        if closest_branch is None:
            closest_branch = -1
        # print("closeAT and closeBranch", closest_At, closest_branch)
        self.clAT.set(closest_At)
        self.clBR.set(closest_branch)

    def onClose(self) -> None:
        super().onClose()
        self.updateOp.unsubscribeToAllGlobalUpdates(
            ReefTrackingAgentBase.OBSERVATIONPOSTFIX, self.__handleReefUpdate
        )
        self.updateOp.unsubscribeToAllGlobalUpdates(
            ObjectLocalizingAgentBase.DETECTIONPOSTFIX, self.__handleObjectUpdate
        )

    def getProbabilityMapDetectionsProtobuf(
        self,
    ) -> XTableValue.ProbabilityMappingDetections:

        probability_map_detections = XTableValue.ProbabilityMappingDetections()

        # TODO: Hook up w/ Robot, Algae and Coral Detections later
        robot_detection_lst: List[XTableValue.RobotDetection] = []
        algae_detection_lst: List[XTableValue.AlgaeDetection] = []
        coral_detection_lst: List[XTableValue.CoralDetection] = []
        reef_state: XTableValue.XTablesValues.ReefState = (
            self.central.reefState.getReefMapState_as_protobuf(self.team)
        )

        probability_map_detections.robots.extend(robot_detection_lst)
        probability_map_detections.algaes.extend(algae_detection_lst)
        probability_map_detections.corals.extend(coral_detection_lst)
        probability_map_detections.reef.CopyFrom(reef_state)
        return probability_map_detections

    def getDescription(self):
        return "Central-Process-Accumulate-Results-Broadcast-Them"

    def isRunning(self):
        return True
