from PyQt5.QtGui import QShowEvent, QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import pyqtSignal


class BaseMainWindow(QMainWindow):
    show_signal = pyqtSignal()

    def __init__(self, title="PartSeg", signal_fun=None):
        super().__init__()
        if signal_fun is not None:
            self.show_signal.connect(signal_fun)
        self.files_num = 1
        self.setAcceptDrops(True)
        self.setWindowTitle(title)
        self.title_base = title

    def showEvent(self, a0: QShowEvent):
        self.show_signal.emit()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def read_drop(self, paths):
        raise NotImplementedError()

    def dropEvent(self, event: QDropEvent):
        assert all([x.isLocalFile() for x in event.mimeData().urls()])
        paths = [x.path() for x in event.mimeData().urls()]
        if self.files_num != -1 and len(paths) > self.files_num:
            QMessageBox.information(self, "To many files", "currently support only drag and drop one file")
            return
        self.read_drop(paths)
