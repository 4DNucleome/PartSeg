from typing import List, Optional, Type

from qtpy.QtGui import QShowEvent, QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import QMainWindow, QMessageBox, QWidget
from qtpy.QtCore import Signal
import os

from PartSeg.common_gui.about_dialog import AboutDialog
from PartSeg.common_gui.image_adjustment import ImageAdjustmentDialog
from PartSeg.common_gui.show_directory_dialog import DirectoryDialog
from PartSeg.common_backend.load_backup import import_config
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog
from PartSegCore.io_utils import ProjectInfoBase
from PartSegImage import Image
from PartSeg.common_backend.base_settings import BaseSettings, SwapTimeStackException, TimeAndStackException


class BaseMainMenu(QWidget):
    def __init__(self, settings: BaseSettings, main_window):
        super().__init__()
        self.settings = settings
        self.main_window = main_window

    def set_data(self, data):
        if isinstance(data, list):
            if len(data) == 0:
                QMessageBox.warning(self, "Empty list", "List of files to load is empty")
                return
            if hasattr(self.main_window, "multiple_files"):
                self.main_window.multiple_files.add_states(data)
                self.main_window.multiple_files.setVisible(True)
                self.settings.set("multiple_files", True)
            data = data[0]
        if isinstance(data, ProjectInfoBase):
            if data.errors != "":
                resp = QMessageBox.question(
                    self,
                    "Load problem",
                    f"During load data "
                    f"some problems occur: {data.errors}."
                    "Do you would like to try load it anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if resp == QMessageBox.No:
                    return
            try:
                image = self._settings.verify_image(data.image, False)
            except SwapTimeStackException:
                res = QMessageBox.question(
                    self,
                    "Not supported",
                    "Time data are currently not supported. Maybe You would like to treat time as z-stack",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )

                if res == QMessageBox.Yes:
                    image = data.image.swap_time_and_stack()
                else:
                    return
            except TimeAndStackException:
                QMessageBox.warning(self, "image error", "Do not support time and stack image")
                return
            if image:
                if isinstance(image, Image):
                    # noinspection PyProtectedMember
                    data = data._replace(image=image)
            else:
                return
        if data is None:
            QMessageBox().warning(self, "Data load fail", "Fail with loading data", QMessageBox.Ok)
            return
        self.settings.set_project_info(data)


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

    def __init__(
        self,
        config_folder: Optional[str] = None,
        title="PartSeg",
        settings: Optional[BaseSettings] = None,
        signal_fun=None,
    ):
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
                errors_message.setInformativeText(
                    "During load saved state some of data could not be load properly\n"
                    "The files has prepared backup copies in "
                    " state directory (Help > State directory)"
                )
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

    def dragEnterEvent(self, event: QDragEnterEvent):  # pylint: disable=R0201
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def read_drop(self, paths: List[str]):
        """Function to process loading files by drag and drop."""
        raise NotImplementedError()

    def _read_drop(self, paths, load_module):
        ext_set = set([os.path.splitext(x)[1] for x in paths])

        def exception_hook(exception):
            if isinstance(exception, OSError):
                QMessageBox().warning(
                    self, "IO Error", "Disc operation error: " + ", ".join(exception.args), QMessageBox.Ok
                )

        for load_class in load_module.load_dict.values():
            if load_class.partial() or load_class.number_of_files() != len(paths):
                continue
            if ext_set.issubset(load_class.get_extensions()):
                dial = ExecuteFunctionDialog(load_class.load, [paths], exception_hook=exception_hook)
                if dial.exec():
                    self.main_menu.set_data(dial.get_result())
                return
        QMessageBox.information(self, "No method", f"No  methods for load files: " + ",".join(paths))

    def dropEvent(self, event: QDropEvent):
        """
        Support for load files by drag and drop.
        At beginning it check number of files and if it greater than :py:attr:`.files_num` it refuse loading. Otherwise
        it call :py:meth:`.read_drop` method and this method should be overwritten in sub classes
        """
        if not all([x.isLocalFile() for x in event.mimeData().urls()]):
            QMessageBox().warning(self, "Load error", "Not all files are locally. Cannot load data.", QMessageBox.Ok)
        paths = [x.toLocalFile() for x in event.mimeData().urls()]
        if self.files_num != -1 and len(paths) > self.files_num:
            QMessageBox.information(self, "To many files", "currently support only drag and drop one file")
            return
        self.read_drop(paths)

    def show_settings_directory(self):
        DirectoryDialog(
            self.settings.json_folder_path, "Path to place where PartSeg store the data between runs"
        ).exec()

    @staticmethod
    def show_about_dialog():
        """Show about dialog."""
        AboutDialog().exec()

    @staticmethod
    def get_project_info(file_path, image):
        raise NotADirectoryError()

    def image_adjust_exec(self):
        dial = ImageAdjustmentDialog(self.settings.image)
        if dial.exec():
            algorithm = dial.result_val.algorithm
            dial2 = ExecuteFunctionDialog(
                algorithm.transform, [], {"image": self.settings.image, "arguments": dial.result_val.values}
            )
            if dial2.exec():
                result: Image = dial2.get_result()
                self.settings.set_project_info(self.get_project_info(result.file_path, result))
