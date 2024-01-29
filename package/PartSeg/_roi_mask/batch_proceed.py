import os
import re
from functools import partial
from pathlib import Path
from queue import Queue
from typing import List, NamedTuple, Optional, Tuple, Union, cast

from pydantic import BaseModel
from qtpy.QtCore import QThread, Signal

from PartSeg._roi_mask.stack_settings import StackSettings, get_mask
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.mask.algorithm_description import MaskAlgorithmSelection
from PartSegCore.mask.io_functions import LoadROIImage, LoadStackImage, MaskProjectTuple, SaveROI
from PartSegCore.segmentation import StackAlgorithm
from PartSegCore.segmentation.algorithm_base import ROIExtractionAlgorithm


class BatchTask(NamedTuple):
    data: Union[str, MaskProjectTuple]
    parameters: ROIExtractionProfile
    save_prefix: Optional[Tuple[Union[str, Path], Union[dict, BaseModel]]]


class BatchProceed(QThread):
    error_signal = Signal(str)
    progress_signal = Signal(str, int, str, int)
    range_signal = Signal(int, int)
    execution_done = Signal()
    multiple_result = Signal(MaskProjectTuple)
    algorithm: ROIExtractionAlgorithm

    def __init__(self):
        super().__init__()
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

    def progress_info(self, name: str, text: str, num: int):
        self.progress_signal.emit(text, num, name, self.index)

    def run_calculation(self):  # noqa: PLR0912  # FIXME
        while not self.queue.empty():
            task: BatchTask = self.queue.get()
            if isinstance(task.data, str):
                file_path = task.data
                if os.path.splitext(task.data)[1] in LoadROIImage.get_extensions():
                    project_tuple = LoadROIImage.load([task.data])
                else:
                    project_tuple = LoadStackImage.load([task.data])
            elif isinstance(task.data, MaskProjectTuple):
                project_tuple: MaskProjectTuple = task.data
                file_path = project_tuple.image.file_path
            else:
                continue
            try:
                name = os.path.basename(file_path)
                blank = get_mask(project_tuple.roi_info.roi, project_tuple.mask, project_tuple.selected_components)
                algorithm = cast(StackAlgorithm, MaskAlgorithmSelection[task.parameters.algorithm]())
                algorithm.set_image(project_tuple.image)
                algorithm.set_mask(blank)
                algorithm.set_parameters(task.parameters.values)
                if isinstance(task.save_prefix, tuple):
                    self.range_signal.emit(0, algorithm.get_steps_num() + 1)
                else:
                    self.range_signal.emit(0, algorithm.get_steps_num())
                # noinspection PyTypeChecker
                segmentation = algorithm.calculation_run(partial(self.progress_info, name))
                state2 = StackSettings.transform_state(
                    project_tuple,
                    segmentation.roi_info,
                    (
                        {i: segmentation.parameters for i in segmentation.roi_info.bound_info}
                        if segmentation.roi_info is not None
                        else {}
                    ),
                    [],
                )
                if isinstance(task.save_prefix, tuple):
                    self.progress_info(name, "saving", algorithm.get_steps_num())
                    name = f"{os.path.splitext(os.path.basename(file_path))[0]}.seg"
                    re_end = re.compile(r"(.*_version)(\d+)\.seg$")
                    os.makedirs(task.save_prefix[0], exist_ok=True)
                    while os.path.exists(os.path.join(task.save_prefix[0], name)):
                        if match := re_end.match(name):
                            num = int(match[2]) + 1
                            name = match[1] + str(num) + ".seg"
                        else:
                            name = f"{os.path.splitext(os.path.basename(file_path))[0]}_version1.seg"
                    SaveROI.save(os.path.join(task.save_prefix[0], name), state2, parameters=task.save_prefix[1])
                else:
                    self.multiple_result.emit(state2)
            except Exception as e:  # pylint: disable=broad-except
                self.error_signal.emit(f"Exception occurred during proceed {file_path}. Exception info {e}")
            self.index += 1
        self.index = 0
        self.execution_done.emit()

    def run(self):
        self.index = 0
        self.run_calculation()
