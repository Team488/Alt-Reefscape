from Core.Neo import Neo
from Core.Agents import FrameDisplayer

if __name__ == "__main":
    n = Neo()
    n.wakeAgent(FrameDisplayer, isMainThread=True)
    n.shutDown()
