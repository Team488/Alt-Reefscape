from Core.Neo import Neo
from Core.Agents.PathToNearestCoralStation import PathToNearestCoralStation
from Core.Agents.PathToNearestBarge import PathToNearestBarge

if __name__ == "__main__":

    # import signal
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    #
    n = Neo()
    #
    # n.wakeAgent(CentralAgent, isMainThread=False)
    n.wakeAgent(PathToNearestCoralStation, isMainThread=False)
    n.wakeAgent(PathToNearestBarge, isMainThread=False)

    #
    # object_localization = ObjectLocalizingAgentPartial(
    #     inferenceMode=InferenceMode.ALCOROBEST2025GPUONLY,
    #     capture=D435Capture(res=D435IResolution.RS720P),
    #     cameraExtrinsics=ColorCameraExtrinsics2025.DEPTH_REAR_LEFT,
    # )

    # n.wakeAgent(object_localization, isMainThread=False)

    # n.wakeAgent(orinIngestorAgent,isMainThread=False)
    # n.wakeAgent(
    #     partialVideoWriterAgent(FileCapture(0), savePath=f"orinCam_{getTimeStr()}.mp4"),
    #     isMainThread=False,
    # )

    # # start pathplanning rpc
    # from pathplanning.nmc import fastMarchingMethodRPC
    #
    # fastMarchingMethodRPC.serve()
