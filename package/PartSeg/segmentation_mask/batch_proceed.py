import re
from collections import defaultdict
from functools import partial
from os import path
from typing import Type, Optional, NamedTuple, Union, Tuple, List
from queue import Queue

from qtpy.QtCore import QThread, Signal

from PartSeg.segmentation_mask.stack_settings import get_mask, StackSettings
from PartSeg.utils.algorithm_describe_base import SegmentationProfile
from PartSeg.utils.mask.algorithm_description import mask_algorithm_dict
from PartSeg.utils.mask.io_functions import SaveSegmentation, LoadSegmentationImage, LoadImage, SegmentationTuple
from ..utils.segmentation.algorithm_base import SegmentationAlgorithm


class BatchTask(NamedTuple):
    data: Union[str, SegmentationTuple]
    parameters: SegmentationProfile
    save_prefix: Optional[Tuple[str, dict]]


class BatchProceed(QThread):
    error_signal = Signal(str)
    progress_signal = Signal(str, int,  str, int)
    range_signal = Signal(int, int)
    execution_done = Signal()
    multiple_result = Signal(SegmentationTuple)
    algorithm: SegmentationAlgorithm

    def __init__(self):
        super(BatchProceed, self).__init__()
        self.queue = Queue()
        self.algorithm = Optional[None]
        self.parameters = None
        self.file_list = []
        self.index = 0
        self.result_dir = ""
        self.save_parameters = {}

    def add_task(self, task: Union[BatchTask, List[BatchTask]]):
        if isinstance(task, list):
            for el in task:
                self.queue.put(el)
        else:
            self.queue.put(task)

    def progress_info(self, name, text, num):
        self.progress_signal.emit(text, num, name, self.index)

    def run_calculation(self):
        while not self.queue.empty():
            task: BatchTask = self.queue.get()
            if isinstance(task.data, str):
                file_path = task.data
                if path.splitext(task.data)[1] == ".seg":
                    project_tuple = LoadSegmentationImage.load([task.data])
                else:
                    project_tuple = LoadImage.load([task.data])
            elif isinstance(task.data, SegmentationTuple):
                project_tuple: SegmentationTuple = task.data
                file_path = project_tuple.image.file_path
            else:
                continue
            try:
                name = path.basename(file_path)
                blank = get_mask(project_tuple.segmentation, project_tuple.chosen_components)
                algorithm = mask_algorithm_dict[task.parameters.algorithm]()
                algorithm.set_image(project_tuple.image)
                algorithm.set_mask(blank)
                algorithm.set_parameters(**task.parameters.values)
                if isinstance(task.save_prefix, tuple):
                    self.range_signal.emit(0, algorithm.get_steps_num() + 1)
                else:
                    self.range_signal.emit(0, algorithm.get_steps_num())
                segmentation = algorithm.calculation_run(partial(self.progress_info, name))
                state2 = StackSettings.transform_state(project_tuple, segmentation.segmentation,
                                                       defaultdict(lambda: segmentation.parameters), [])
                if isinstance(task.save_prefix, tuple):
                    self.progress_info(name, "saving", algorithm.get_steps_num())
                    name = path.splitext(path.basename(file_path))[0] + ".seg"
                    re_end = re.compile(r"(.*_version)(\d+)\.seg$")
                    while path.exists(path.join(task.save_prefix[0], name)):
                        match = re_end.match(name)
                        if match:
                            num = int(match.group(2)) + 1
                            name = match.group(1) + str(num) + ".seg"
                        else:
                            name = path.splitext(path.basename(file_path))[0] + "_version1.seg"
                    SaveSegmentation.save(path.join(task.save_prefix[0], name), state2,
                                          parameters=task.save_prefix[1])
                else:
                    self.multiple_result.emit(state2)
            except Exception as e:
                self.error_signal.emit("Exception occurred during proceed {}. Exception info {}".format(file_path, e))
            self.index += 1
        self.index = 0
        self.execution_done.emit()

    def run(self):
        self.index = 0
        self.run_calculation()
