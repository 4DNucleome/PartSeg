import logging
import time
from collections import namedtuple
from uuid import uuid4

from partseg_old.batch_processing.parallel_backed import BatchManager

Test = namedtuple("Test", ["x", "y", "z"])

logging.basicConfig(level=logging.DEBUG)


class Tester(object):
    def __init__(self, suffix):
        self.uuid = uuid4()
        self.suffix = suffix


def calc_fun(name, data):
    time.sleep(0.1)
    return name + data.suffix

input_data = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent feugiat mauris posuere sem mollis, suscipit malesuada magna placerat. Proin semper tristique neque, id laoreet quam. Phasellus tristique metus nec libero posuere, in finibus ligula dignissim. Quisque rhoncus metus purus. Mauris tempor elementum enim id scelerisque. Vivamus id lacinia nisl, ut rhoncus nunc. Praesent in lacinia sem, sed rhoncus nulla.

Integer semper ac risus quis finibus. Nulla vitae risus at massa finibus fringilla. Nam efficitur nisi lorem. Mauris mi sapien, pellentesque id tellus commodo, facilisis facilisis turpis. Etiam fringilla sed mauris at ullamcorper. Fusce tincidunt est a urna condimentum, a condimentum lacus eleifend. Cras at justo ut diam vulputate euismod. Fusce convallis viverra congue. Maecenas eu nulla a nunc aliquet fermentum sit amet eu odio.""".split()

global_data = Tester("_aaa")
global_data2 = Tester("_bbb")

manager = BatchManager()
manager.set_number_off_process(7)
manager.add_work(input_data, global_data, calc_fun)
manager.add_work(input_data, global_data2, calc_fun)
while manager.has_work:
    res = manager.get_result()
    if len(res) != 0:
        print(res)
    time.sleep(0.1)
print (manager.process_list)
time.sleep(5)

manager.set_number_off_process(1)
time.sleep(1)
print("Buka")
time.sleep(1)
manager.set_number_off_process(0)

while not manager.finished:
    print("test")
    time.sleep(0.1)
