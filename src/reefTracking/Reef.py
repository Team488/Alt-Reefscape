from typing import Dict, List, Tuple
from enum import Enum
import math

# Define the missing constants
RED_ALLIANCE_TAGS = [1, 2, 3, 4, 5, 6, 7, 8]
BLUE_ALLIANCE_TAGS = [11, 12, 13, 14, 15, 16, 17, 18]
BRANCHES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


class Alliance(Enum):
    RED = 0, "Red"
    BLUE = 1, "Blue"


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


class Reef:
    class Branch(Enum):
        A = 0, "A"
        B = 1, "B"
        C = 2, "C"
        D = 3, "D"
        E = 4, "E"
        F = 5, "F"
        G = 6, "G"
        H = 7, "H"
        I = 8, "I"
        J = 9, "J"
        K = 10, "K"
        L = 11, "L"

    class ReefFace(Enum):
        # Indexes for ALLIANCE_TAGS
        CLOSE = 0
        CLOSE_LEFT = 5
        CLOSE_RIGHT = 1
        FAR = 2
        FAR_LEFT = 3
        FAR_RIGHT = 4

    class CoralState(Enum):
        OFF = 0
        ON = 1

    class Level(Enum):
        L2 = 0, "L2"
        L3 = 1, "L3"
        L4 = 2, "L4"

    global BLUE_ALLIANCE_TAGS
    global RED_ALLIANCE_TAGS
    global BRANCHES

    BLUE_ALLIANCE_TAGS = [18, 17, 22, 21, 20, 19]
    RED_ALLIANCE_TAGS = [7, 8, 9, 10, 11, 6]
    REEF_FACE_NOTATION = ["CLOSE"]

    BRANCHES = [branch for branch in Branch]
    # TAG INDEXES [A, B, C, D, E, F, G, H, I, J, K, L]

    def __init__(self, alliance: Alliance) -> None:
        self.alliance = alliance
        assert (
            self.alliance == Alliance.RED or self.alliance == Alliance.BLUE
        ), "Alliance Not Properly Initialized"

        # Initialize
        self.init_alliance_settings()
        self.init_branch_to_tag()
        self.init_branch_states()

    def init_alliance_settings(self) -> None:
        # Initialize Alliance Specific Settings
        if self.alliance == Alliance.RED:
            self.alliance_tags = RED_ALLIANCE_TAGS
        elif self.alliance == Alliance.BLUE:
            self.alliance_tags = BLUE_ALLIANCE_TAGS

    def init_branch_to_tag(self) -> None:
        # Branch Char -> Tag Dictionary
        self.branch_to_tag = {}
        index = 0
        char_index = 0
        for branch in BRANCHES:
            self.branch_to_tag.update({branch: self.alliance_tags[index]})
            char_index += 1
            if char_index % 2 == 0:
                index += 1

    def init_branch_states(self) -> None:
        # Initialize Branch States:
        self.branch_state: Dict[str, Dict["Reef.Level", "Reef.CoralState"]] = {}
        for branch in BRANCHES:
            # Initialize L2, L3, L4
            self.branch_state.update(
                {
                    branch: {
                        self.Level.L2: self.CoralState.OFF,
                        self.Level.L3: self.CoralState.OFF,
                        self.Level.L4: self.CoralState.OFF,
                    }
                }
            )

    def get_all_states(self) -> Dict[str, Dict["Reef.Level", "Reef.CoralState"]]:
        return self.branch_state

    # Get the state of the branches
    def get_branch(
        self, branch: "Reef.Branch"
    ) -> Dict["Reef.Level", "Reef.CoralState"]:
        return self.branch_state.get(branch.name, {})

    # get_branch_state_at("A", Reef.Level.L2) => Reef.BranchState.OFF
    def get_branch_state_at(
        self, branch: "Reef.Branch", level: "Reef.Level"
    ) -> "Reef.CoralState":
        branch_face = self.get_branch(branch)
        # print("branch", branch_face)
        return branch_face.get(level, self.CoralState.OFF)

    # get_branches_at_tag(7) => ["A", "B"]
    def get_branches_at_tag(self, id: int) -> List[str]:
        if id in self.alliance_tags:
            index = self.alliance_tags.index(id) * 2
            return BRANCHES[index : index + 2]
        return []

    def get_self_alliance_tags(self) -> List[int]:
        return self.alliance_tags

    # get_tag_from_branch("A") => 7
    def get_tag_from_branch(self, branch: "Reef.Branch") -> int:
        index = int(math.floor(branch.value[0] / 2))  # Retrieves the index
        return self.alliance_tags[index]

    # toggle_branch(Reef.Branch.A, Reef.Level.L1) => sets to true
    def set_branch_state(self, branch: Branch, level: Level, state: CoralState) -> None:
        self.branch_state[branch.name][level] = state

    # get_branch_with_state(CoralState.ON) => [A, B, C] which contains only CoralState.ON
    def get_branch_with_state(self, state: "Reef.CoralState") -> List[Tuple[str, str]]:
        lst: List[Tuple[str, str]] = []
        for branch_str, levels in self.branch_state.items():
            for level, coral_state in levels.items():
                if coral_state == state:
                    lst.append((branch_str, level.name))
        return lst

    # id : 6 => ReefFace.CLOSE_LEFT
    def get_tag_to_ReefFace(self, tag_id: int) -> "Reef.ReefFace":
        index = self.alliance_tags.index(tag_id)
        return self.ReefFace(index)

    def get_ReefFace_to_tag(self, reef_face: "Reef.ReefFace") -> int:
        return self.alliance_tags[reef_face.value]

    # returns the branch and their states at level level
    def get_branches_at_level_with_state(
        self, level: "Reef.Level", state: "Reef.CoralState"
    ) -> List[str]:
        return [
            branch_str
            for branch_str, levels in self.branch_state.items()
            if levels[level] == state
        ]

    def printBranchList(self) -> None:
        print(BRANCHES)


red = Reef(Alliance.RED)
blue = Reef(Alliance.BLUE)


# print("=======RED======")
"""
for x in range(6, 12):
    print(x, red.get_tag_to_ReefFace(x))
"""
# print(red.get_ReefFace_to_tag(Reef.ReefFace.FAR_LEFT))

# print("=======BLUE======")
# for x in range(17, 22):
#    print(x, blue.get_branches_at_tag(x))
# print(test.get_branch_state_at('A', Reef.Level.L2))
# print(red.get_all_states())

# for x in Reef.Branch:
#    print(x, red.get_tag_from_branch(x))

# robot.goToBranch(L1, l2)
# goToReefBranch(A = Branch, L1 = Level)
"""
print(red.get_branch_state_at(Reef.Branch.A, Reef.Level.L2))
red.set_branch_state(Reef.Branch.A, Reef.Level.L2, Reef.CoralState.ON)
print(red.get_branch_state_at(Reef.Branch.A, Reef.Level.L2))
print(red.get_all_states())
print("CORALS ON", red.get_branch_with_state(Reef.CoralState.ON))
print("CORALS OFF", red.get_branch_with_state(Reef.CoralState.OFF))
print("=====================")

print(red.get_branches_at_level_with_state(Reef.Level.L2, Reef.CoralState.OFF))
"""
