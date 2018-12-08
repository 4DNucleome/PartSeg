from PyQt5.QtCore import QThread, pyqtSignal


class ProgressTread(QThread):
    range_changed = pyqtSignal(int, int)
    step_changed = pyqtSignal(int)
    error_signal = pyqtSignal(Exception)

    def info_function(self, label: str, val: int):
        if label == "max":
            self.range_changed.emit(0, val)
        elif label == "step":
            self.step_changed.emit(val)
