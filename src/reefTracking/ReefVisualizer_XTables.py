import time
import random
from kivy.clock import Clock
from threading import Thread
from ReefVisualizer import ReefVisualizer
import Reef
from functools import partial

from abstract.Agent import Agent


class ReefVisualizerAgent(Agent):
    def __init__(self, **kwargs) -> None:
        self.visualizer = kwargs.get()

    def create(self) -> None:
        super().create()

        self.branch_states = []

        self.colors = {
            "red": (1, 0, 0, 1),
            "green": (0, 1, 0, 1),
            "yellow": (1, 1, 0, 1),
        }

        self.xclient.subscribe(
            "REEFMAP_STATES", consumer=lambda ret: self.__updateVisualizer
        )
        self.branch_idx = ["L2, L3, L4"]
        self.reef = Reef(Reef.Alliance.BLUE)

        self.visualizer_map_render = {}

    def runPeriodic(self) -> None:
        super().runPeriodic()
        # self.runVisualizer()

    def runVisualizer(self) -> None:
        for key, value in self.visualizer_map_render.items():
            self.visualizer.queue_color_update(key, value)  # queue and update the UI

    def __updateVisualizer(self, ret) -> None:
        # list[tuple[int,int,float]]
        # List w/ tuple (April tag id, branch id, openness confidence)
        self.branch_states = ret.value

        for atID_x, branchID_x, confidence in self.branch_states:
            branch_index = branchID_x % 2
            branch_level = branchID_x // 2

            branches = self.reef.get_branches_at_tag(
                atID_x
            )  # This returns 17 => [A, B]
            branch = branches[branch_index]  # Branch Object
            branch_name = branch.value[1]

            level_name = self.branch_idx[branch_level]
            branch_level_index_str = f"{branch_name}_{level_name}"  # "A_L1"

            if (
                confidence < 0.8
                and self.visualizer_map_render[branch_level_index_str]
                is not self.colors["red"]
            ):
                self.visualizer_map_render[branch_level_index_str] = self.colors["red"]
            else:  # Filled up
                self.visualizer_map_render[branch_level_index_str] = self.colors[
                    "green"
                ]

    def onClose(self) -> None:
        super().onClose()
        for key in self.keys:
            self.xclient.unsubscribe(
                self.getDetectionTable(key), consumer=self.__handleObjectUpdate
            )


def ReefVisualizerAgentPartial(visualizer):
    """Returns a partially completed ReefVisualizerAgent. All you have to do is pass it into neo"""
    return partial(ReefVisualizer, visualizer=visualizer)
