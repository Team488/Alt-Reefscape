import time
import random
from kivy.clock import Clock
from threading import Thread
from reefTracking.ReefVisualizer import ReefVisualizerApp
from reefTracking import ReefVisualizer
from reefTracking.Reef import Reef, Alliance
from tools.Constants import TEAM
from functools import partial

from abstract.Agent import Agent
from coreinterface.ReefPacket import ReefPacket


class ReefVisualizerAgent(Agent):
    def __init__(self, **kwargs) -> None:
        self.visualizer = kwargs.get("visualizer")
        self.reef = Reef(Alliance.BLUE)
        self.branch_idx = ["L2", "L3", "L4"]
        self.visualizer_map_render = {}

    def create(self) -> None:
        super().create()

        self.branch_states = []

        self.colors = {
            "red": (1, 0, 0, 1),
            "green": (0, 1, 0, 1),
            "yellow": (1, 1, 0, 1),
            "gray": (0.3, 0.3, 0.3, 1),
        }

        self.xclient.subscribe(
            "REEFMAP_STATES", consumer=lambda ret: self.__updateVisualizer(ret)
        )
        team = self.getTeam()
        if team == TEAM.RED:
            self.reef = Reef(Alliance.RED)
        elif team == TEAM.BLUE:
            self.reef = Reef(Alliance.BLUE)
        else:
            raise Exception("Invalid Team Reading from XTables")

    def runPeriodic(self) -> None:
        super().runPeriodic()

    def runVisualizer(self):
        if self.visualizer is not None:
            for key, value in self.visualizer_map_render.items():
                self.visualizer.queue_color_update(key, value)  # queue and update the U
        else:
            print(self.visualizer, " is None")

    def __updateVisualizer(self, ret) -> None:
        # list[tuple[int,int,float]]
        # List w/ tuple (April tag id, branch id, openness confidence)
        bytes = ret.value
        decoded = ReefPacket.fromBytes(bytes)
        flattenedCoralOutput = ReefPacket.getFlattenedObservations(decoded)[0]

        for atID_x, branchID_x, confidence in flattenedCoralOutput:
            # print(atID_x, branchID_x, confidence)
            if confidence != 0.5:
                print(atID_x, branchID_x, confidence)
            branch_index = branchID_x % 2  # Index Left or Right side of AT (0, 1)
            branch_level = branchID_x // 2  # Index for L2, L3, L4 (0, 1, 2)

            branches = self.reef.get_branches_at_tag(
                atID_x
            )  # This returns 17 => [A, B]
            branch = branches[branch_index]  # Branch Object
            branch_name = branch.value[1]

            level_name = self.branch_idx[branch_level]
            branch_level_index_str = f"{branch_name}_{level_name}"  # "A_L1"

            # subject to change
            if 0.0 <= confidence < 0.3:
                self.visualizer_map_render[branch_level_index_str] = self.colors["red"]
            elif 0.3 <= confidence < 0.7:
                self.visualizer_map_render[branch_level_index_str] = self.colors[
                    "yellow"
                ]
            elif 0.7 <= confidence:
                self.visualizer_map_render[branch_level_index_str] = self.colors[
                    "green"
                ]
            else:
                print("confidence error...", confidence)
                self.visualizer_map_render[branch_level_index_str] = self.colors["gray"]

        print("Updating Visualizer:")
        self.runVisualizer()

    def getDescription(self) -> str:
        """Sooo, what do you do for a "living" """
        return "Visuals for Reef Tracking"

    def isRunning(self) -> bool:
        """Return a boolean value denoting whether the agent should still be running"""
        return True

    def onClose(self) -> None:
        super().onClose()
        self.xclient.unsubscribe("REEFMAP_STATES", consumer=self.__updateVisualizer)


def ReefVisualizerAgentPartial(visualizer):
    """Returns a partially completed ReefVisualizerAgent. All you have to do is pass it into neo"""
    return partial(ReefVisualizerAgent, visualizer=visualizer)
