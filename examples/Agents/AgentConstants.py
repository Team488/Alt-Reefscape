from typing import Union, List
from functools import partial
from enum import Enum
from abstract.Capture import Capture
from abstract.Agent import Agent


class AgentCapabilites(Enum):

    stream = "stream_queue"

    @property
    def objectName(self):
        return self.value

    @staticmethod
    def getCapabilites(
        agentClass: Union[partial, type[Agent]]
    ) -> List["AgentCapabilites"]:
        if isinstance(agentClass, partial):
            return AgentCapabilites.__getPartialCapabilites(agentClass)

        return AgentCapabilites.__getAgentCapabilites(agentClass)

    @staticmethod
    def __getPartialCapabilites(agentClass: partial) -> List["AgentCapabilites"]:
        from Core.Agents.Abstract.CameraUsingAgentBase import CameraUsingAgentBase

        # TODO much more here
        capabilites = set()
        for arg in agentClass.args:
            if issubclass(type(arg), Capture):
                capabilites.add(AgentCapabilites.stream)

        for arg in agentClass.keywords.values():
            if issubclass(type(arg), Capture):
                capabilites.add(AgentCapabilites.stream)

        if issubclass(agentClass.func, CameraUsingAgentBase):
            capabilites.add(AgentCapabilites.stream)

        return list(capabilites)

    @staticmethod
    def __getAgentCapabilites(agentClass: type[Agent]) -> List["AgentCapabilites"]:
        from Core.Agents.Abstract.CameraUsingAgentBase import CameraUsingAgentBase

        # TODO much more here
        capabilites = set()
        if issubclass(agentClass, CameraUsingAgentBase):
            capabilites.add(AgentCapabilites.stream)

        return list(capabilites)
