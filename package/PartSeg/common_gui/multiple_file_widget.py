import os
import sys
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List

from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtGui import QFontMetrics, QMouseEvent, QResizeEvent
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QInputDialog,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from PartSeg.common_backend.base_settings import BaseSettings
from PartSegCore.io_utils import LoadBase
from PartSegCore.project_info import ProjectInfoBase

from .custom_load_dialog import CustomLoadDialog, LoadProperty
from .waiting_dialog import ExecuteFunctionDialog


class CustomTreeWidget(QTreeWidget):
    context_load = Signal(QTreeWidgetItem)
    context_compare = Signal(QTreeWidgetItem)
    context_forget = Signal(QTreeWidgetItem)

    def __init__(self, compare, parent=None):
        super().__init__(parent)
        self.compare = compare
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)

    def showContextMenu(self, point):
        item = self.itemAt(point)
        if item is None:
            return
        menu = QMenu()
        if item.parent() is not None:
            action1 = QAction("Load")
            action1.triggered.connect(partial(self.context_load.emit, item))
            menu.addAction(action1)
            if self.compare and item.text(0) not in ["raw image", "image with mask"]:
                action2 = QAction("Compare")
                action2.triggered.connect(partial(self.context_compare.emit, item))
                menu.addAction(action2)
        action = QAction("Forget")
        action.triggered.connect(partial(self.context_forget.emit, item))
        menu.addAction(action)
        menu.exec_(self.mapToGlobal(point))

    def set_show_compare(self, compare: bool):
        self.compare = compare

    def mouseMoveEvent(self, _):  # pylint: disable=R0201
        QApplication.setOverrideCursor(Qt.ArrowCursor)


class MultipleFileWidget(QWidget):
    _add_state = Signal(object, bool)

    def __init__(self, settings: BaseSettings, load_dict: Dict[str, LoadBase], compare_in_context_menu=False):
        super().__init__()
        self.settings = settings
        self.state_dict: Dict[str, Dict[str, ProjectInfoBase]] = defaultdict(dict)
        self.state_dict_count = Counter()
        self.file_list = []
        self.load_register = load_dict
        self.file_view = CustomTreeWidget(compare_in_context_menu)
        self.file_view.header().close()
        self.save_state_btn = QPushButton("Save state")
        self.save_state_btn.setStyleSheet("QPushButton{font-weight: bold;}")
        self.load_files_btn = QPushButton("Load Files")
        self.forget_btn = QPushButton("Forget")

        self.save_state_btn.clicked.connect(self.save_state)
        self.forget_btn.clicked.connect(self.forget)
        self.load_files_btn.clicked.connect(self.load_files)
        self.file_view.itemDoubleClicked.connect(self.load_state)
        self.file_view.context_load.connect(self.load_state)
        self.last_point = None

        self.custom_names_chk = QCheckBox("Custom names")

        layout = QGridLayout()
        layout.addWidget(self.file_view, 0, 0, 1, 2)
        layout.addWidget(self.save_state_btn, 1, 0, 1, 2)
        layout.addWidget(self.load_files_btn, 2, 0)
        layout.addWidget(self.forget_btn, 2, 1)
        layout.addWidget(self.custom_names_chk, 3, 0, 1, 2)

        self.setLayout(layout)
        self.setMouseTracking(True)
        self.file_view.setMouseTracking(True)
        self.file_view.context_load.connect(self.load_state)
        self.file_view.context_compare.connect(self.load_compare)
        self.file_view.context_forget.connect(self.forget_action)
        self.error_list = []

        self._add_state.connect(self.save_state_action)
        self.settings.data_changed.connect(self.view_changed)

    def view_changed(self, path, value):
        if path == "multiple_files_widget":
            self.setVisible(value)

    def execute_load_files(self, load_data: LoadProperty, range_changed, step_changed):
        range_changed(0, len(load_data.load_location))
        for i, el in enumerate(load_data.load_location, 1):
            load_list = [el]
            while load_data.load_class.number_of_files() > len(load_list):
                load_list.append(load_data.load_class.get_next_file(load_list))
                if not os.path.exists(load_list[-1]):
                    self.error_list.append(el)
                    step_changed(i)
                    continue
            state: ProjectInfoBase = load_data.load_class.load(load_list)
            self._add_state.emit(state, False)
            step_changed(i)

    def load_files(self):
        def exception_hook(exception):
            from qtpy.QtCore import QMetaObject

            instance = QApplication.instance()
            if isinstance(exception, MemoryError):
                instance.warning = "Open error", f"Not enough memory to read this image: {exception}"
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
            elif isinstance(exception, IOError):
                instance.warning = "Open error", f"Some problem with reading from disc: {exception}"
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
            elif isinstance(exception, KeyError):
                instance.warning = "Open error", f"Some problem project file: {exception}"
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
                print(exception, file=sys.stderr)
            else:
                raise exception

        dial = MultipleLoadDialog(self.load_register, self.settings.get_path_history())
        dial.setDirectory(self.settings.get("io.multiple_open_directory", str(Path.home())))
        dial.selectNameFilter(self.settings.get("io.multiple_open_filter", next(iter(self.load_register.keys()))))
        self.error_list = []
        if dial.exec():
            result = dial.get_result()
            load_dir = os.path.dirname(result.load_location[0])
            self.settings.set("io.multiple_open_directory", load_dir)
            self.settings.add_path_history(load_dir)
            self.settings.set("io.multiple_open_filter", result.selected_filter)

            dial_fun = ExecuteFunctionDialog(self.execute_load_files, [result], exception_hook=exception_hook)
            dial_fun.exec()
            if self.error_list:
                errors_message = QMessageBox()
                errors_message.setText("There are errors during load files")
                errors_message.setInformativeText("During load files cannot found some of files on disc")
                errors_message.setStandardButtons(QMessageBox.Ok)
                text = "\n".join(["File: " + x[0] + "\n" + str(x[1]) for x in self.error_list])
                errors_message.setDetailedText(text)
                errors_message.exec()

    def load_state(self, item, _column=1):
        if item.parent() is None:
            return
        file_name = self.file_list[self.file_view.indexOfTopLevelItem(item.parent())]
        state_name = item.text(0)
        project_info = self.state_dict[file_name][state_name]
        try:
            self.parent().parent().parent().set_data(project_info)
        except AttributeError:
            self.settings.set_project_info(project_info)

    def load_compare(self, item):
        if item.parent() is None:
            return
        file_name = self.file_list[self.file_view.indexOfTopLevelItem(item.parent())]
        if self.settings.image.file_path != file_name:
            QMessageBox.information(self, "Wrong file", "Please select same file as main")
            return
        state_name = item.text(0)
        project_info = self.state_dict[file_name][state_name]
        if hasattr(self.settings, "set_segmentation_to_compare"):
            self.settings.set_segmentation_to_compare(project_info.roi_info)

    def save_state(self):
        state: ProjectInfoBase = self.settings.get_project_info()
        custom_name = self.custom_names_chk.isChecked()
        self.save_state_action(state, custom_name)

    def save_state_action(self, state: ProjectInfoBase, custom_name):
        # TODO left elipsis
        # state: ProjectInfoBase = self.get_state()
        normed_file_path = os.path.normpath(state.file_path)
        sub_dict = self.state_dict[normed_file_path]
        name = f"state {self.state_dict_count[normed_file_path]+1}"
        if custom_name:
            name, ok = QInputDialog.getText(self, "Save name", "Save name:", text=name)
            if not ok:
                return
            while name in sub_dict or name in ["raw image", "image with mask"]:
                name, ok = QInputDialog.getText(self, "Save name", "Save name (previous in use):", text=name)
                if not ok:
                    return
        try:
            index = self.file_list.index(os.path.normpath(normed_file_path))
            item = self.file_view.topLevelItem(index)
        except ValueError:
            metric = QFontMetrics(self.file_view.font())
            width = self.file_view.width() - 45
            clipped_text = metric.elidedText(normed_file_path, Qt.ElideLeft, width)
            item = QTreeWidgetItem(self.file_view, [clipped_text])
            item.setToolTip(0, normed_file_path)
            self.file_list.append(normed_file_path)
            QTreeWidgetItem(item, ["raw image"])
            sub_dict["raw image"] = state.get_raw_copy()
            if state.is_masked():
                QTreeWidgetItem(item, ["image with mask"])
                sub_dict["image with mask"] = state.get_raw_mask_copy()

        item.setExpanded(True)
        if state.is_raw():
            return
        it = QTreeWidgetItem(item, [name])
        self.file_view.setCurrentItem(it)
        sub_dict[name] = state
        self.state_dict_count[state.file_path] += 1

    def forget(self):
        if not self.forget_btn.isEnabled():
            return
        self.forget_btn.setDisabled(True)
        item: QTreeWidgetItem = self.file_view.currentItem()
        self.forget_action(item)

    def forget_action(self, item):
        if item is None:
            return
        if isinstance(item.parent(), QTreeWidgetItem):
            index = self.file_view.indexOfTopLevelItem(item.parent())
            text = self.file_list[index]
            if item.text(0) not in self.state_dict[text]:
                return
            del self.state_dict[text][item.text(0)]
            parent = item.parent()
            parent.removeChild(item)
            if parent.childCount() == 0:
                self.file_view.takeTopLevelItem(index)
                self.file_list.remove(text)

        else:
            index = self.file_view.indexOfTopLevelItem(item)
            text = self.file_list[index]
            del self.state_dict[text]
            del self.state_dict_count[text]
            self.file_list.remove(text)
            self.file_view.takeTopLevelItem(index)
        QTimer().singleShot(500, self.enable_forget)

    @Slot()
    def enable_forget(self):
        self.forget_btn.setEnabled(True)

    def resizeEvent(self, event: QResizeEvent):
        metric = QFontMetrics(self.file_view.font())
        width = self.file_view.width() - 45
        for i, text in enumerate(self.file_list):
            clipped_text = metric.elidedText(text, Qt.ElideLeft, width)
            item: QTreeWidgetItem = self.file_view.topLevelItem(i)
            item.setText(0, clipped_text)

    def mousePressEvent(self, event: QMouseEvent):
        if event.x() > self.width() - 20:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.x() > self.width() - 20:
            QApplication.setOverrideCursor(Qt.SplitHCursor)
        else:
            QApplication.setOverrideCursor(Qt.ArrowCursor)
        if self.last_point is None or not (event.buttons() & Qt.LeftButton):
            return
        new_width = event.x() + 10
        new_width = max(new_width, 150)
        new_width = min(new_width, 600)

        self.setMinimumWidth(new_width)

    def leaveEvent(self, _):  # pylint: disable=R0201
        QApplication.setOverrideCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.last_point = None

    def set_compare_in_context_menu(self, compare: bool):
        self.file_view.set_show_compare(compare)

    def add_states(self, states: List[ProjectInfoBase]):
        """add multiple states to widget"""
        for el in states:
            self.save_state_action(el, False)


class MultipleLoadDialog(CustomLoadDialog):
    def __init__(self, load_register, history=None):
        load_register = {key: val for key, val in load_register.items() if not val.partial()}
        super().__init__(load_register=load_register, history=history)
        self.setFileMode(QFileDialog.ExistingFiles)

    def accept(self):
        self.files_list.extend(self.selectedFiles())
        QFileDialog.accept(self)
