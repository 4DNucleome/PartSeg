import sys

from qtpy.QtCore import QMutex

from ..utils.segmentation.algorithm_base import SegmentationAlgorithm, SegmentationResult
from qtpy.QtCore import QThread, Signal


class SegmentationThread(QThread):
    execution_done = Signal(SegmentationResult)
    progress_signal = Signal(str, int)
    info_signal = Signal(str)
    exception_occurred = Signal(Exception)

    def __init__(self, algorithm: SegmentationAlgorithm):
        super().__init__()
        self.finished.connect(self.finished_task)
        self.algorithm = algorithm
        self.clean_later = False
        self.cache = None
        self.mutex = QMutex()
        self.rerun = False, QThread.InheritPriority

    def get_info_text(self):
        return self.algorithm.get_info_text()

    def send_info(self, text, num):
        self.progress_signal.emit(text, num)

    def run(self):
        if self.algorithm.image is None:
            # assertion for running algorithm without image
            print(f"No image in class {self.algorithm.__class__}", file=sys.stderr)
            return
        try:
            segment_data = self.algorithm.calculation_run(self.send_info)
        except Exception as e:
            self.exception_occurred.emit(e)
            return
        if segment_data is None:
            return
        self.execution_done.emit(segment_data)

    def finished_task(self):
        self.mutex.lock()
        if self.cache is not None:
            args, kwargs = self.cache
            self.algorithm.set_parameters(*args, **kwargs)
            self.cache = None
            self.clean_later = False
        if self.rerun[0]:
            self.rerun = False, QThread.InheritPriority
            super().start(self.rerun[1])
        elif self.clean_later:
            self.algorithm._clean()
            self.clean_later = False
        self.mutex.unlock()

    def clean(self):
        self.mutex.lock()
        if self.isRunning():
            self.clean_later = True
        else:
            self.algorithm._clean()
        self.mutex.unlock()

    def set_parameters(self, *args, **kwargs):
        self.mutex.lock()
        if self.isRunning():
            self.cache = args, kwargs
            self.clean_later = False
        else:
            self.algorithm.set_parameters(*args, **kwargs)
        self.mutex.unlock()

    def start(self, priority: 'QThread.Priority' = QThread.InheritPriority):
        self.mutex.lock()
        if self.isRunning():
            self.clean_later = False
            self.rerun = True, priority
        else:
            super().start(priority)
        self.mutex.unlock()
