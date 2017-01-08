from parallel_backed import BatchManager
import logging
from collections import namedtuple
import time


Test = namedtuple("Test", ["x", "y", "z"])

logging.basicConfig(level=logging.DEBUG)


manager = BatchManager()
manager.set_number_off_process(7)
manager.add_work(["buka", {"ala": 7, "marysia": 10}, Test(7, 8, 12), "a", "b"] + list(range(100)))
print (manager.process_list)
time.sleep(2)

manager.set_number_off_process(1)
time.sleep(1)
print("Buka")
time.sleep(1)
manager.set_number_off_process(0)

while not manager.finished:
    print("test")
    time.sleep(0.1)
