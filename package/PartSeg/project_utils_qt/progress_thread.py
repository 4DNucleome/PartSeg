from qtpy.QtCore import QThread, Signal


class ProgressTread(QThread):
    range_changed = Signal(int, int)
    step_changed = Signal(int)
    error_signal = Signal(Exception)

    def info_function(self, label: str, val: int):
        if label == "max":
            self.range_changed.emit(0, val)
        elif label == "step":
            self.step_changed.emit(val)
