import json
from typing import Dict, Optional, List

import numpy as np
from JXTABLES import XTableValues_pb2 as XTableValue

from Alt.Core import getChildLogger
from Alt.Core.Constants.Teams import TEAM
from Alt.Core.Units.Types import Length

from ..Constants.AprilTags import ATLocations
from .Reef import ReefBranches
from .ReefPacket import ReefPacket, reefStatePacket_capnp

Sentinel = getChildLogger("Reef_State")


class ReefState:
    def __init__(self, DISSIPATIONFACTOR=0.999) -> None:
        (
            self.idx_flip,
            self.idx_to_apriltag,
            self.apriltag_to_idx,
            self.reef_map,
        ) = self.__createReefMap()
        self.algae_map = self.__createAlgaeMap()
        self.DISSIPATIONFACTOR = DISSIPATIONFACTOR

    def __createReefMap(self) -> tuple[int, list[int], Dict[int, int], np.ndarray]:
        idx_to_apriltag_blue = ATLocations.getReefBasedIds(TEAM.BLUE)
        idx_to_apriltag_red = ATLocations.getReefBasedIds(TEAM.RED)
        idx_flip = len(idx_to_apriltag_blue)  # index of team flip
        idx_to_apriltag = idx_to_apriltag_blue + idx_to_apriltag_red
        apriltag_to_idx = {}
        for idx, apriltag in enumerate(idx_to_apriltag):
            apriltag_to_idx[apriltag] = idx

        cols = len(idx_to_apriltag)
        rows = len(ReefBranches)
        reef_map = np.full(
            (rows, cols), 0.5, dtype=np.float64
        )  # Initialize to 50% as "unknown"

        return idx_flip, idx_to_apriltag, apriltag_to_idx, reef_map

    def __createAlgaeMap(self):
        idx_to_apriltag = ATLocations.getReefBasedIds()
        numAlgae = len(idx_to_apriltag)
        algae_map = np.zeros((numAlgae), dtype=np.float64)
        return algae_map

    def dissipateOverTime(self, timeFactor: int) -> None:
        """This operates under the assumtion that as time passed, the chance of slots being taken increases. So if there are no updates, as t grows, the openness confidence should go to zero"""

        """ Reef Map Dissipation"""
        coral_dissipation_factor = np.power(
            self.DISSIPATIONFACTOR, round(timeFactor / 5)
        )

        # Create a mask for slots that are not locked (not -1) and above 0.5 to dissipate
        # That way we can keep "unknown" states
        mask = (self.reef_map != -1) & (self.reef_map > 0.5)

        # Apply the new dissipation
        new_coral_map = (
            0.5 + (self.reef_map[mask] - 0.5) * coral_dissipation_factor
        )  # discrete dissipation

        # Epislon Threshold to reset detections approach 0.5
        epsilon = 1e-1
        new_coral_map[np.abs(new_coral_map - 0.5) < epsilon] = 0.5

        # Update the reef map:
        self.reef_map[mask] = new_coral_map

        """ Algae Map Dissipation"""
        algae_dissipation_factor = np.power(
            self.DISSIPATIONFACTOR, round(timeFactor / 10)
        )
        self.algae_map *= algae_dissipation_factor  # discrete dissipation

    def addObservationCoral(
        self, apriltagid, branchid, opennessconfidence, weighingfactor=0.85
    ) -> None:
        if apriltagid not in self.apriltag_to_idx or (
            branchid < 0 or branchid >= self.reef_map.shape[0]
        ):
            Sentinel.warning(
                f"Invalid apriltagid or branchid! {apriltagid=} {branchid=}"
            )
            return

        col_idx = self.apriltag_to_idx.get(apriltagid)
        row_idx = branchid
        # print("Added observation", col_idx, row_idx)

        # We know 100% that the space is filled.
        # Stop updating to that particular observation. It becomes "locked".
        # TODO: Locking Mechanism Commented Out. Add it in if necessary

        if self.reef_map[row_idx, col_idx] < 0.1:
            self.reef_map[row_idx, col_idx] = -1.0
            return

        self.reef_map[row_idx, col_idx] *= 1 - weighingfactor
        self.reef_map[row_idx, col_idx] += opennessconfidence * weighingfactor

    def addObservationAlgae(
        self, apriltagid, opennessconfidence, weighingfactor=0.85
    ) -> None:
        if apriltagid not in self.apriltag_to_idx:
            Sentinel.warning(f"Invalid apriltagid{apriltagid=}")
            return

        algae_idx = self.apriltag_to_idx.get(apriltagid)

        # Locking mechanism for the algae (we only care when it's off)
        if self.algae_map[algae_idx] > 0.95:
            self.algae_map[algae_idx] = 1.0

        # Skip if locked
        if self.algae_map[algae_idx] == 1.0:
            return

        self.algae_map[algae_idx] *= 1 - weighingfactor
        self.algae_map[algae_idx] += opennessconfidence * weighingfactor

    def getOpenSlotsAboveT(
        self,
        team: TEAM = None,
        threshold=0.5,
        algaeThreshold=0.7,
        considerAlgaeBlocking=True,
    ) -> list[tuple[int, int, float]]:
        """Returns open slots in the form of a tuple with (April tag id, branch id, openness confidence)"""
        offset_col, mapbacking = self.__getMapBacking(team)
        row_idxs, col_idxs = np.where(mapbacking > threshold)

        open_slots = []
        for row, col in zip(row_idxs, col_idxs):
            at_idx = self.idx_to_apriltag[col + offset_col]
            branch_idx = row

            blockedBranchIdxs = ATLocations.getBlockedBranchIdxs(at_idx)
            algaeOccupancy = self.algae_map[self.apriltag_to_idx[at_idx]]

            openness = mapbacking[row, col]

            # if we are checking for algae blocking and the branch is one that could be blocked,
            # if we meet an algae occupance threshold ignore it
            if considerAlgaeBlocking and (
                algaeOccupancy > algaeThreshold and branch_idx in blockedBranchIdxs
            ):
                Sentinel.debug(f"Blocked algae at {at_idx=} {row=} {blockedBranchIdxs}")
                continue

            open_slots.append((int(at_idx), int(branch_idx), float(openness)))

        return open_slots

    def getHighestSlot(self, team: TEAM = None) -> Optional[tuple[int, int, float]]:
        """Returns open slots in the form of a tuple with (April tag id, branch id, openness confidence)"""
        offset_col, mapbacking = self.__getMapBacking(team)
        if not (mapbacking > 0).any():
            return None

        max = np.argmax(mapbacking)
        row, col = np.unravel_index(max, mapbacking.shape)
        branch_idx = row
        at_idx = self.idx_to_apriltag[col + offset_col]
        openness = mapbacking[row, col]

        return at_idx, branch_idx, openness

    # Helper
    def getReefMapState_as_dictionary(
        self, team: TEAM = None
    ) -> dict[(int, int):float]:
        """Returns the entire map state as a dictionary"""
        offset_col, mapbacking = self.__getMapBacking(team)
        reefMap_state = {}
        rows, cols = mapbacking.shape
        for col in range(cols):
            for row in range(rows):
                at_id = self.idx_to_apriltag[col + offset_col]
                openness = mapbacking[row, col]
                reefMap_state[(int(at_id), int(row))] = float(openness)

        return reefMap_state

    def getReefMapState_as_ReefPacket(
        self, team: TEAM = None, timestamp=0
    ) -> reefStatePacket_capnp.ReefPacket:
        # Create the Coral Map Output
        offset_col, mapbacking = self.__getMapBacking(team)
        coralTrackerOutput = {}
        rows, cols = mapbacking.shape
        for col in range(cols):
            for row in range(rows):
                at_id = self.idx_to_apriltag[col + offset_col]
                openness = mapbacking[row, col]
                if at_id not in coralTrackerOutput:
                    coralTrackerOutput[at_id] = {}
                coralTrackerOutput[at_id][row] = openness
        # Create the Algae Map Output
        algaeTrackerOutput = {}
        for apriltag in self.idx_to_apriltag:
            algae_idx = self.apriltag_to_idx.get(apriltag)
            algaeTrackerOutput[apriltag] = self.algae_map[algae_idx]

        message = "Reef State Update"
        return ReefPacket.createPacket(
            coralTrackerOutput, algaeTrackerOutput, message, timestamp
        )

    def getReefMapState_as_protobuf(
        self, team: TEAM = None, timestamp=0
    ) -> XTableValue.ReefState:
        reef_state_proto = XTableValue.ReefState()
        reef_state_entries: List[XTableValue.ReefEntry] = []

        # Create the Coral Map Output
        offset_col, mapbacking = self.__getMapBacking(team)
        rows, cols = mapbacking.shape
        for col in range(cols):
            at_id = self.idx_to_apriltag[col + offset_col]
            reef_entry = XTableValue.ReefEntry()
            branch_coral_state_lst: List[XTableValue.BranchCoralState] = []
            for row in range(rows):
                openness = mapbacking[row, col]
                branch_coral_state = XTableValue.BranchCoralState()
                branch_coral_state.index = row + 1
                branch_coral_state.openness = openness
                branch_coral_state_lst.append(branch_coral_state)
            reef_entry.aprilTagID = at_id
            reef_entry.algaeOpenness = self.apriltag_to_idx.get(at_id)
            reef_entry.branchIndexStates.extend(branch_coral_state_lst)

            reef_state_entries.append(reef_entry)

        reef_state_proto.entries.extend(reef_state_entries)
        return reef_state_proto

    def getReefMapState_as_Json(
        self, team: TEAM = None, timestamp=0
    ) -> reefStatePacket_capnp.ReefPacket:
        # Create the Coral Map Output
        offset_col, mapbacking = self.__getMapBacking(team)
        coralTrackerOutput = {}
        rows, cols = mapbacking.shape
        for col in range(cols):
            for row in range(rows):
                at_id = self.idx_to_apriltag[col + offset_col]
                openness = mapbacking[row, col]
                if at_id not in coralTrackerOutput:
                    coralTrackerOutput[at_id] = {}
                coralTrackerOutput[at_id][row] = openness
        # Create the Algae Map Output
        algaeTrackerOutput = {}
        for apriltag in self.idx_to_apriltag:
            algae_idx = self.apriltag_to_idx.get(apriltag)
            algaeTrackerOutput[apriltag] = self.algae_map[algae_idx]

        jsonstr = json.dumps((coralTrackerOutput, algaeTrackerOutput))
        return jsonstr

    def __getMapBacking(self, team: TEAM):
        mapbacking = self.reef_map
        offset_col = 0
        if team is not None:
            if team == TEAM.BLUE:
                mapbacking = mapbacking[:, : self.idx_flip]
            elif team == TEAM.RED:
                mapbacking = mapbacking[:, self.idx_flip :]
                offset_col = self.idx_flip
        return offset_col, mapbacking

    def __getDist(self, robotPos2CMRAd: tuple[float, float, float], atId: int):
        atPoseXYCM = ATLocations.get_pose_by_id(atId, length=Length.CM)[0][:2]
        robotXYCm = robotPos2CMRAd[:2]
        return np.linalg.norm((np.subtract(atPoseXYCM, robotXYCm)))

    def getClosestOpen(
        self,
        robotPos2CMRAd: tuple[float, float, float],
        team: TEAM = None,
        threshold=0.5,
        algaeThreshold=0.7,
        considerAlgaeBlocking=True,
    ):
        open_slots = self.getOpenSlotsAboveT(
            team, threshold, algaeThreshold, considerAlgaeBlocking
        )
        closestAT = None
        closestBranch = None
        closestDist = 1e6
        for atid, branchid, _ in open_slots:
            if closestAT != atid:
                # skip same tag if already checked that one as the closest
                d = self.__getDist(robotPos2CMRAd, atid)
                if d < closestDist:
                    closestAT = atid
                    closestBranch = branchid
                    closestDist = d

        return closestAT, closestBranch
