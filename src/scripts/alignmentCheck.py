from Core.Agents.Partials.AlignmentProviderAgent import partialAlignmentCheck
from tools.Constants import SimulationEndpoints, CommonVideos
from Alignment.ReefPostAlignmentProviderBlobs import ReefPostAlignmentProvider
from Core.Neo import Neo

if __name__ == "__main__":
    alignmentCheckReefLeft = partialAlignmentCheck(
        alignmentProvider=ReefPostAlignmentProvider(),
        # cameraPath=CommonVideos.StingerCam.path,
        cameraPath=0,
        showFrames=True,
        # flushCamMs=50,
    )

    n = Neo()

    n.wakeAgent(alignmentCheckReefLeft, isMainThread=True)

    n.waitForAgentsFinished()

    n.shutDown()
