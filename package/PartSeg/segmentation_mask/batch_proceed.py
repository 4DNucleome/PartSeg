import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .io_functions import load_stack_segmentation, SaveSegmentation, SegmentationTuple
from ..partseg_utils.segmentation.algorithm_base import SegmentationAlgorithm
from ..project_utils_qt.settings import ImageSettings
from PartSeg.tiff_image import ImageReader
from typing import Type
from os import path


class BatchProceed(QThread):
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str, int)
    execution_done = pyqtSignal()
    algorithm: SegmentationAlgorithm

    def __init__(self):
        super(BatchProceed, self).__init__()
        self.algorithm = None
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
        temp_settings = ImageSettings()
        reader = ImageReader()
        while self.index < len(self.file_list):
            file_path = self.file_list[self.index]
            try:
                if path.splitext(file_path)[1] == ".seg":
                    segmentation, metadata = load_stack_segmentation(file_path)
                    if "base_file" not in metadata or not path.exists(metadata["base_file"]):
                        self.index += 1
                        self.error_signal.emit("not found base file for {}".format(file_path))
                        continue
                    self.base_file = metadata["base_file"]
                    self.components = metadata["components"]
                    if len(self.components) > 250:
                        blank = np.zeros(segmentation.shape, dtype=np.uint16)
                    else:
                        blank = np.zeros(segmentation.shape, dtype=np.uint8)
                    for i, v in enumerate(self.components):
                        blank[segmentation == v] = i + 1
                else:
                    self.base_file = file_path
                    self.components = []
                    blank = None
                temp_settings.image = reader.read(self.base_file)
                self.algorithm.set_parameters(image=temp_settings.image.get_channel(self.channel_num),
                                              exclude_mask=blank, **self.parameters)
                segmentation = self.algorithm.calculation_run(self.progress_info)
                name = path.basename(file_path)
                name = path.splitext(name)[0] + ".seg"
                SaveSegmentation.save(path.join(self.result_dir, name),
                                      SegmentationTuple(temp_settings.image, segmentation.segmentation,
                                                        list(range(1, len(self.components) + 1))),
                                      parameters=SaveSegmentation.get_default_values())

            except Exception as e:
                self.error_signal.emit("Exception occurred during proceed {}. Exception info {}".format(file_path, e))
            self.index += 1
        if self.index >= len(self.file_list):
            self.execution_done.emit()

    def run(self):
        self.index = 0
        self.run_calculation()
