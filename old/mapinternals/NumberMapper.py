from typing import List, Optional


class NumberMapper:
    def __init__(
        self, redrobot_numbers: List[str], bluerobot_numbers: List[str]
    ) -> None:
        self.rednums = redrobot_numbers
        self.bluenums = bluerobot_numbers

    def getRobotNumberEstimate(self, colorisblue: bool, numberestimate: str) -> str:
        if len(numberestimate) == 0:
            print("Empty number estimate")
            return ""
        nums = self.bluenums if colorisblue else self.rednums
        best = ""
        best_num_overlaps = 0
        for num in nums:
            num_overlaps = 0
            i = 0
            minlen = min(len(num), len(numberestimate))
            while i < minlen:
                if num[i] == numberestimate[i]:
                    num_overlaps += 1
                i += 1
            if num_overlaps > best_num_overlaps:
                best_num_overlaps = num_overlaps
                best = num

        return best
