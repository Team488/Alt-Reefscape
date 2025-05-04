from abc import ABC, abstractmethod


class Demo(ABC):
    @abstractmethod
    def startDemo(self) -> None:
        # every demo will have a start demo
        pass
