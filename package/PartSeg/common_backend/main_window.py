from typing import List, Optional, Type

from qtpy.QtGui import QShowEvent, QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import QMainWindow, QMessageBox
from qtpy.QtCore import Signal
import os

from PartSeg.common_gui.about_dialog import AboutDialog
from PartSeg.common_gui.show_directory_dialog import DirectoryDialog
from PartSeg.common_backend.load_backup import import_config
from .base_settings import BaseSettings


class BaseMainWindow(QMainWindow):
    """
    Base for main windows of subprograms

    :ivar settings: store state of application. initial value is obtained from :py:attr:`.settings_class`
    :ivar files_num: maximal number of files accepted by drag and rop event
    :param config_folder: path to directory in which application save state. If `settings` parameter is note
        then settings object is created with passing this path to :py:attr:`.settings_class`.
        If this parameter and `settings`
        are None then constructor fail with :py:exc:`ValueError`.
    :param title: Window default title
    :param settings: object to store application state
    :param signal_fun: function which need to be called when window shown.
    """
    show_signal = Signal()
    """Signal emitted when window has shown. Used to hide Launcher."""

    @classmethod
    def get_setting_class(cls) -> Type[BaseSettings]:
        """Get constructor for :py:attr:`settings`"""
        return BaseSettings

    def __init__(self, config_folder: Optional[str] = None, title="PartSeg", settings: Optional[BaseSettings] = None,
                 signal_fun=None):
        if settings is None:
            if config_folder is None:
                raise ValueError("wrong config folder")
            settings: BaseSettings = self.get_setting_class()(config_folder)
            if not os.path.exists(config_folder):
                import_config()
            errors = settings.load()
            if errors:
                errors_message = QMessageBox()
                errors_message.setText("There are errors during start")
                errors_message.setInformativeText("During load saved state some of data could not be load properly\n"
                                                  "The files has prepared backup copies in "
                                                  " state directory (Help > State directory)")
                errors_message.setStandardButtons(QMessageBox.Ok)
                text = "\n".join(["File: " + x[0] + "\n" + str(x[1]) for x in errors])
                errors_message.setDetailedText(text)
                errors_message.exec()

        super().__init__()
        if signal_fun is not None:
            self.show_signal.connect(signal_fun)
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

    def read_drop(self, paths: List[str]):
        """Function to process loading files by drag and drop."""
        raise NotImplementedError()

    def dropEvent(self, event: QDropEvent):
        """
        Support for load files by drag and drop.
        At beginning it check number of files and if it greater than :py:attr:`.files_num` it refuse loading. Otherwise
        it call :py:meth:`.read_drop` method and this method should be overwritten in sub classes
        """
        assert all([x.isLocalFile() for x in event.mimeData().urls()])
        paths = [x.toLocalFile() for x in event.mimeData().urls()]
        if self.files_num != -1 and len(paths) > self.files_num:
            QMessageBox.information(self, "To many files", "currently support only drag and drop one file")
            return
        self.read_drop(paths)

    def show_settings_directory(self):
        DirectoryDialog(
            self.settings.json_folder_path, "Path to place where PartSeg store the data between runs").exec()

    @staticmethod
    def show_about_dialog():
        """Show about dialog."""
        AboutDialog().exec()
