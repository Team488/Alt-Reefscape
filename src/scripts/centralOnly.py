from Core.Agents.Abstract import ObjectLocalizingAgentPartial
from Core.Agents.CentralAgent import CentralAgent
from Core.Neo import Neo
from Core.Agents.PathToNearestCoralStation import PathToNearestCoralStation
from tools.Constants import InferenceMode, D435IResolution, ColorCameraExtrinsics2025
from Captures import D435Capture

if __name__ == "__main":
    n = Neo()
    n.wakeAgent(CentralAgent, isMainThread=True)
    n.shutDown()
