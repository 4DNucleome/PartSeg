from collections import defaultdict

from qtpy.QtCore import QThread, Signal

from PartSeg.segmentation_mask.stack_settings import get_mask, StackSettings
from PartSeg.utils.mask.io_functions import SaveSegmentation, SegmentationTuple, \
    LoadSegmentationImage, LoadImage
from ..utils.segmentation.algorithm_base import SegmentationAlgorithm
from typing import Type, Optional
from os import path
import re


class BatchProceed(QThread):
    error_signal = Signal(str)
    progress_signal = Signal(str, int)
    execution_done = Signal()
    algorithm: SegmentationAlgorithm

    def __init__(self):
        super(BatchProceed, self).__init__()
        self.algorithm = Optional[None]
        self.parameters = None
        self.file_list = []
        self.base_file = ""
        self.components = []
        self.index = 0
        self.result_dir = ""
        self.channel_num = 0

    def set_parameters(self, algorithm: Type[SegmentationAlgorithm], parameters, channel_num, file_list, result_dir):
        self.algorithm = algorithm()
        self.parameters = parameters
        self.file_list = list(sorted(file_list))
        self.result_dir = result_dir
        self.channel_num = channel_num
        # self.algorithm.execution_done.connect(self.calc_one_finished)
        # self.algorithm.progress_signal.connect(self.progress_info)

    def progress_info(self, text, _num):
        name = path.basename(self.file_list[self.index])
        self.progress_signal.emit("file {} ({}): {}".format(self.index+1, name, text), self.index)

    def run_calculation(self):

        while self.index < len(self.file_list):
            file_path = self.file_list[self.index]
            try:
                if path.splitext(file_path)[1] == ".seg":
                    project_tuple = LoadSegmentationImage.load(file_path)
                else:
                    project_tuple = LoadImage.load(file_path)
                blank = get_mask(project_tuple.segmentation, project_tuple.chosen_components)
                self.algorithm.set_parameters(image=project_tuple.image.get_channel(self.channel_num),
                                              exclude_mask=blank, **self.parameters)
                segmentation = self.algorithm.calculation_run(self.progress_info)
                state2 = StackSettings.transform_state(project_tuple, segmentation.segmentation,
                                                       defaultdict(lambda: segmentation.parameters), [])
                name = path.splitext(path.basename(file_path))[0] + ".seg"
                re_end = re.compile(r"(.*_version)(\d+)\.seg$")
                while path.exists(name):
                    match = re_end.match(name)
                    if match:
                        num = int(match.group(2))+1
                        name = match.group(1) + str(num) + ".seg"
                    else:
                        name = path.splitext(path.basename(file_path))[0] + "_version1.seg"

                # FIXME
                SaveSegmentation.save(path.join(self.result_dir, name), state2,
                                      parameters=SaveSegmentation.get_default_values())

            except Exception as e:
                self.error_signal.emit("Exception occurred during proceed {}. Exception info {}".format(file_path, e))
            self.index += 1
        if self.index >= len(self.file_list):
            self.execution_done.emit()

    def run(self):
        self.index = 0
        self.run_calculation()
