from qt_import import QThread, pyqtSignal
import numpy as np


class SegmentationAlgorithm(QThread):
    execution_done = pyqtSignal(np.ndarray)
    progress_signal = pyqtSignal(str, int)
    info_signal = pyqtSignal(str)

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()
