from Alt.Core import Neo

from ..Agents.AlignmentProviderAgent import partialAlignmentProviderAgent
from ..Alignment.ReefPostAlignmentProviderBlobs import ReefPostAlignmentProvider

if __name__ == "__main__":
    alignmentCheckReefLeft = partialAlignmentProviderAgent(
        alignmentProvider=ReefPostAlignmentProvider(),
        cameraPath=0,
        showFrames=True,
    )

    n = Neo()

    n.wakeAgent(alignmentCheckReefLeft, isMainThread=True)

    n.waitForAgentsFinished()

    n.shutDown()
