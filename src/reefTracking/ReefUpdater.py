from .ReefState import ReefState

class ReefUpdater:
    def __init__(
        self,
    ) -> None:
        self.reefState = ReefState()

    def processReefUpdate(
        self,
        reefResults: tuple[list[tuple[int, int, float]], list[tuple[int, float]]],
        timeStepMs,
    ) -> None:
        self.reefState.dissipateOverTime(timeStepMs)

        for reefResult in reefResults:
            # print(reefResult)
            coralObservation, algaeObservation = reefResult
            for apriltagid, branchid, opennessconfidence in coralObservation:
                self.reefState.addObservationCoral(
                    apriltagid, branchid, opennessconfidence
                )

            for apriltagid, opennessconfidence in algaeObservation:
                self.reefState.addObservationAlgae(apriltagid, opennessconfidence)
