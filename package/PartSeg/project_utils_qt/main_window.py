from qtpy.QtGui import QShowEvent, QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import QMainWindow, QMessageBox
from qtpy.QtCore import Signal
import os

from PartSeg.project_utils_qt.load_backup import import_config
from .settings import BaseSettings


class BaseMainWindow(QMainWindow):
    show_signal = Signal()

    settings_class = BaseSettings

    def __init__(self, config_folder=None, title="PartSeg", settings=None, signal_fun=None):
        super().__init__()
        if signal_fun is not None:
            self.show_signal.connect(signal_fun)
        if settings is None:
            if config_folder is None:
                raise ValueError("wrong config folder")
            self.settings = self.settings_class(config_folder)
            if not os.path.exists(config_folder):
                import_config()
            errors = self.settings.load()
            if errors:
                errors_message = QMessageBox()
                errors_message.setText("There are errors during start")
                errors_message.setInformativeText("During load saved state some of data could not be load properly\n"
                                                  "The files has prepared backup copies in  state directory (Help > State directory)")
                errors_message.setStandardButtons(QMessageBox.Ok)
                text = "\n".join(["File: " + x[0] + "\n" + str(x[1]) for x in errors])
                errors_message.setDetailedText(text)
                errors_message.exec()
        else:
            self.settings = settings
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
