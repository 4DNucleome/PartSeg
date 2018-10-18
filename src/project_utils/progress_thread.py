from PyQt5.QtCore import QThread, pyqtSignal

class ProgressTread(QThread):
    range_changed = pyqtSignal(int, int)
    step_changed = pyqtSignal(int)