from Alt.Core import Neo

from ..Agents.AlignmentProviderAgent import AlignmentProviderAgent
from ..Alignment.ReefPostAlignmentProviderBlobs import ReefPostAlignmentProvider

if __name__ == "__main__":
    alignmentCheckReefLeft = AlignmentProviderAgent.bind(
        alignmentProvider=ReefPostAlignmentProvider(),
        cameraPath=0,
        showFrames=True,
    )

    n = Neo()

    n.wakeAgent(alignmentCheckReefLeft, isMainThread=True)

    n.waitForAgentsFinished()

    n.shutDown()
