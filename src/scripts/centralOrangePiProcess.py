from Alignment.ReefPostAlignmentProviderBlobs import ReefPostAlignmentProvider
from Core.Agents.Partials.AlignmentProviderAgent import partialAlignmentCheck
from Core.Neo import Neo

alignment = partialAlignmentCheck(
    alignmentProvider=ReefPostAlignmentProvider(), showFrames=False,cameraPath="/dev/color_camera"
)

if __name__ == "__main__":
    n = Neo()
    n.wakeAgent(alignment, isMainThread=True)
    n.shutDown()

