from PyQt5.QtGui import QShowEvent, QCloseEvent
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import pyqtSignal


class BaseMainWindow(QMainWindow):
    show_signal = pyqtSignal()

    def __init__(self, signal_fun=None):
        super().__init__()
        if signal_fun is not None:
            self.show_signal.connect(signal_fun)

    def showEvent(self, a0: QShowEvent):
        self.show_signal.emit()
