from PyQt5.QtCore import QMutex

from project_utils.algorithm_base import SegmentationAlgorithm
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np

class SegmentationThread(QThread):
    execution_done = pyqtSignal([np.ndarray], [np.ndarray, np.ndarray])
    execution_done_extend = pyqtSignal(np.ndarray, np.ndarray)
    progress_signal = pyqtSignal(str, int)
    info_signal = pyqtSignal(str)
    exception_occurred = pyqtSignal(Exception)

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
        try:
            segment_data = self.algorithm.calculation_run(self.send_info)
        except Exception as e:
            return
        if segment_data is None:
            return
        if isinstance(segment_data, tuple):
            self.execution_done.emit(segment_data[0])
            self.execution_done[np.ndarray, np.ndarray].emit(segment_data[0], segment_data[1])
        else:
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