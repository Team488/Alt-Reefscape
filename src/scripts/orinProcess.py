from multiprocessing import Process

import PathToNearestCoralStation
import PathToNearestBarge

p2 = Process(target=PathToNearestBarge.start)
p1 = Process(target=PathToNearestCoralStation.start)

p1.start()
p2.start()
p1.join()
p2.join()
