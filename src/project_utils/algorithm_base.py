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

    def clean(self):
        self.image = None
        self.segmentation = None

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()
