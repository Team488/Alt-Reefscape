from abstract.Order import Order
from tools import NtUtils


class TargetUpdatingOrder(Order):
    TARGETKEY = "net-target-key"

    def create(self) -> None:
        pass

    def run(self, input) -> None:
        target = NtUtils.getPose2dFromBytes(input)
        self.shareOperator.put(
            TargetUpdatingOrder.TARGETKEY, target[:2]
        )  # x,y part until we use rotation

    def getDescription(self) -> str:
        return "Updates_network_target"

    def getName(self) -> None:
        "target_updating_order"
