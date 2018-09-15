from PyQt5.QtCore import QMutex

from qt_import import QThread, pyqtSignal
import numpy as np


class SegmentationAlgorithm(QThread):
    execution_done = pyqtSignal(np.ndarray)
    progress_signal = pyqtSignal(str, int)
    info_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.image = None
        self.segmentation = None
        self.clean_later = False
        self.mutex = QMutex()
        self.finished.connect(self.finished_task)
        self.cache =None
        self.rerun = False, QThread.InheritPriority

    def finished_task(self):
        self.mutex.lock()
        if self.cache is not None:
            args, kwargs = self.cache
            self.set_parameters(*args, **kwargs)
            self.cache = None
            self.clean_later = False
        if self.rerun[0]:
            self.rerun = False, QThread.InheritPriority
            super().start(self.rerun[1])
        elif self.clean_later:
            self._clean()
            self.clean_later = False

        self.mutex.unlock()

    def clean(self):
        self.mutex.lock()
        if self.isRunning():
            self.clean_later = True
        else:
            self._clean()
        self.mutex.unlock()


    def _clean(self):
        self.image = None
        self.segmentation = None

    def set_parameters_wait(self, *args, **kwargs):
        self.mutex.lock()
        if self.isRunning():
            self.cache = args, kwargs
            self.clean_later = False
        else:
            self.set_parameters(*args, **kwargs)
        self.mutex.unlock()

    def start(self, priority: 'QThread.Priority' = QThread.InheritPriority):
        self.mutex.lock()
        if self.isRunning():
            self.clean_later = False
            self.rerun = True, priority
        else:
            super().start(priority)
        self.mutex.unlock()


    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()
