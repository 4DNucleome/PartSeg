import dataclasses
import os
from pathlib import Path
from typing import List, Optional, Type

from qtpy.QtCore import Signal
from qtpy.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QShowEvent
from qtpy.QtWidgets import QAction, QApplication, QFileDialog, QMainWindow, QMenu, QMessageBox, QWidget
from vispy.color import colormap

from PartSegCore.algorithm_describe_base import Register
from PartSegCore.io_utils import LoadBase, SaveScreenshot
from PartSegCore.project_info import ProjectInfoBase
from PartSegImage import Image

from ..common_backend.base_settings import FILE_HISTORY, BaseSettings, SwapTimeStackException, TimeAndStackException
from ..common_backend.load_backup import import_config
from .about_dialog import AboutDialog
from .custom_save_dialog import SaveDialog
from .image_adjustment import ImageAdjustmentDialog
from .napari_image_view import ImageView
from .napari_viewer_wrap import Viewer
from .show_directory_dialog import DirectoryDialog
from .waiting_dialog import ExecuteFunctionDialog


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
                image = self.settings.verify_image(data.image, False)
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
                    data = dataclasses.replace(data, image=image)
            else:
                return
        if data is None:
            QMessageBox().warning(self, "Data load fail", "Fail with loading data", QMessageBox.Ok)
            return
        self.settings.set_project_info(data)
        if data.file_path:
            self.settings.add_path_history(os.path.dirname(data.file_path))


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
        load_dict: Optional[Register] = None,
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
                text = "\n".join("File: " + x[0] + "\n" + str(x[1]) for x in errors)
                errors_message.setDetailedText(text)
                errors_message.exec()

        super().__init__()
        if signal_fun is not None:
            self.show_signal.connect(signal_fun)
        self.settings = settings
        self._load_dict = load_dict
        self.viewer_list: List[Viewer] = []
        self.files_num = 1
        self.setAcceptDrops(True)
        self.setWindowTitle(title)
        self.title_base = title
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(settings.style_sheet)
        self.settings.theme_changed.connect(self.change_theme)
        self.channel_info = ""
        self.multiple_files = None
        self.settings.request_load_files.connect(self.read_drop)
        self.recent_file_menu = QMenu("Open recent")
        self._refresh_recent(FILE_HISTORY, self.settings.get_last_files())
        self.settings.data_changed.connect(self._refresh_recent)
        self.settings.set_parent(self)

    def _refresh_recent(self, name, value):
        if name != FILE_HISTORY or self._load_dict is None:
            return
        self.recent_file_menu.clear()
        for name_list, method in value:
            action = self.recent_file_menu.addAction(f"{name_list[0]}, {method}")
            action.setData((name_list, method))
            action.triggered.connect(self._load_recent)

    def _load_recent(self):
        sender: QAction = self.sender()
        data = sender.data()
        try:
            method: LoadBase = self._load_dict[data[1]]
            dial = ExecuteFunctionDialog(method.load, [data[0]])
            if dial.exec():
                result = dial.get_result()
                self.main_menu.set_data(result)
                self.settings.add_load_files_history(data[0], method.get_name())
        except KeyError:
            self.read_drop(data[0])

    def toggle_multiple_files(self):
        self.settings.set("multiple_files_widget", not self.settings.get("multiple_files_widget"))

    def get_colormaps(self) -> List[Optional[colormap.Colormap]]:
        channel_num = self.settings.image.channels
        if not self.channel_info:
            return [None for _ in range(channel_num)]
        colormaps_name = [self.settings.get_channel_info(self.channel_info, i) for i in range(channel_num)]
        return [self.settings.colormap_dict[name][0] for name in colormaps_name]

    def napari_viewer_show(self):
        viewer = Viewer(title="Additional output", settings=self.settings, partseg_viewer_name=self.channel_info)
        viewer.theme = self.settings.theme_name
        viewer.create_initial_layers(image=True, roi=True, additional_layers=False, points=True)
        self.viewer_list.append(viewer)
        viewer.window.qt_viewer.destroyed.connect(lambda x: self.close_viewer(viewer))

    def additional_layers_show(self, with_channels=False):
        if not self.settings.additional_layers:
            QMessageBox().information(self, "No data", "Last executed algoritm does not provide additional data")
            return
        viewer = Viewer(title="Additional output", settings=self.settings, partseg_viewer_name=self.channel_info)
        viewer.theme = self.settings.theme_name
        viewer.create_initial_layers(image=with_channels, roi=False, additional_layers=True, points=False)
        self.viewer_list.append(viewer)
        viewer.window.qt_viewer.destroyed.connect(lambda x: self.close_viewer(viewer))

    def close_viewer(self, obj):
        for i, el in enumerate(self.viewer_list):
            if el == obj:
                self.viewer_list.pop(i)
                break

    def change_theme(self):
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(self.settings.style_sheet)

    def showEvent(self, a0: QShowEvent):
        self.show_signal.emit()

    def dragEnterEvent(self, event: QDragEnterEvent):  # pylint: disable=R0201
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def read_drop(self, paths: List[str]):

        """Function to process loading files by drag and drop."""
        self._read_drop(paths, self._load_dict)

    def _read_drop(self, paths, load_dict):
        ext_set = {os.path.splitext(x)[1].lower() for x in paths}

        def exception_hook(exception):
            if isinstance(exception, OSError):
                QMessageBox().warning(
                    self, "IO Error", "Disc operation error: " + ", ".join(exception.args), QMessageBox.Ok
                )

        for load_class in load_dict.values():
            if load_class.partial() or load_class.number_of_files() != len(paths):
                continue
            if ext_set.issubset(load_class.get_extensions()):
                dial = ExecuteFunctionDialog(load_class.load, [paths], exception_hook=exception_hook)
                if dial.exec():
                    result = dial.get_result()
                    self.main_menu.set_data(result)
                    self.settings.add_load_files_history(paths, load_class.get_name())
                return
        QMessageBox.information(self, "No method", "No methods for load files: " + ",".join(paths))

    def dropEvent(self, event: QDropEvent):
        """
        Support for load files by drag and drop.
        At beginning it check number of files and if it greater than :py:attr:`.files_num` it refuse loading. Otherwise
        it call :py:meth:`.read_drop` method and this method should be overwritten in sub classes
        """
        if not all(x.isLocalFile() for x in event.mimeData().urls()):
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
        AboutDialog().exec_()

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

    def closeEvent(self, event: QCloseEvent):
        for el in self.viewer_list:
            el.close()
            del el
        super().closeEvent(event)

    def screenshot(self, viewer: ImageView):
        def _screenshot():
            data = viewer.viewer_widget.screenshot()
            dial = SaveDialog(
                {SaveScreenshot.get_name(): SaveScreenshot},
                history=self.settings.get_path_history(),
                system_widget=False,
            )
            dial.setFileMode(QFileDialog.AnyFile)
            dial.setDirectory(self.settings.get("io.save_screenshot", str(Path.home())))
            if not dial.exec_():
                return
            res = dial.get_result()
            save_dir = os.path.dirname(str(res.save_destination))
            self.settings.add_path_history(save_dir)
            self.settings.set("io.save_screenshot", str(save_dir))
            res.save_class.save(res.save_destination, data, res.parameters)

        return _screenshot

    def image_read(self):
        self.setWindowTitle(f"{self.title_base}: {os.path.basename(self.settings.image_path)}")
        self.statusBar().showMessage(self.settings.image_path)
