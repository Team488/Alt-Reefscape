from Core.Neo import Neo
from Core.Agents import CalibrationController

if __name__ == "__main":
    n = Neo()
    n.wakeAgent(CalibrationController, isMainThread=True)
    n.shutDown()
