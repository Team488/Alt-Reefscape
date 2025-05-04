from time import localtime, strftime


def getTimeStr(time = None):
    if time is None:
        return strftime("%Y-%m-%d_%H-%M-%S", localtime())
    return strftime("%Y-%m-%d_%H-%M-%S", time)
