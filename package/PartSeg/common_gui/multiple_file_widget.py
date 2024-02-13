import os
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

from qtpy.QtCore import Qt, QTimer, Signal, Slot
from qtpy.QtGui import QFontMetrics, QMouseEvent, QResizeEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui.custom_load_dialog import CustomLoadDialog, LoadProperty
from PartSeg.common_gui.exception_hooks import load_data_exception_hook
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog
from PartSegCore.io_utils import LoadBase
from PartSegCore.project_info import ProjectInfoBase


class MultipleFilesTreeWidget(QTreeWidget):
    context_load = Signal(QTreeWidgetItem)
    context_compare = Signal(QTreeWidgetItem)
    context_forget = Signal(QTreeWidgetItem)

    def __init__(self, compare, parent=None):
        super().__init__(parent)
        self.compare = compare
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
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

    def mouseMoveEvent(self, event):  # pylint: disable=no-self-use
        QApplication.setOverrideCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)


class LoadRecentFiles(QDialog):
    def __init__(self, settings: BaseSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.cancel_btn = QPushButton("Cancel", clicked=self.reject)
        self.load_btn = QPushButton("Load", clicked=self.accept)

        for name_list, method in settings.get_last_files_multiple():
            entry = f"{name_list[0]} {method}"
            item = QListWidgetItem(entry, self.file_list)
            item.setData(Qt.ItemDataRole.UserRole, (name_list, method))

        last_set = {(tuple(x), y) for x, y in settings.get_last_files_multiple()}
        for name_list, method in settings.get_last_files():
            if (tuple(name_list), method) in last_set:
                continue
            entry = f"{name_list[0]} {method}"
            item = QListWidgetItem(entry, self.file_list)
            item.setData(Qt.ItemDataRole.UserRole, (name_list, method))

        layout = QGridLayout()
        layout.addWidget(QLabel("Select files"))
        layout.addWidget(self.file_list, 1, 0, 1, 2)
        layout.addWidget(self.cancel_btn, 2, 0)
        layout.addWidget(self.load_btn, 2, 1)

        self.setLayout(layout)
        self.resize(
            *self.settings.get_from_profile("multiple_files_dialog_size", (self.size().width(), self.size().height()))
        )

    def get_files(self) -> List[Tuple[List[str], str]]:
        return [item.data(Qt.ItemDataRole.UserRole) for item in self.file_list.selectedItems()]

    def accept(self) -> None:
        self.settings.set_in_profile("multiple_files_dialog_size", (self.size().width(), self.size().height()))
        super().accept()


class MultipleFileWidget(QWidget):
    _add_state = Signal(object, bool)

    def __init__(self, settings: BaseSettings, load_dict: Dict[str, LoadBase], compare_in_context_menu=False):
        super().__init__()
        self.settings = settings
        self.state_dict: Dict[str, Dict[str, ProjectInfoBase]] = defaultdict(dict)
        self.state_dict_count = Counter()
        self.file_list = []
        self.load_register = load_dict
        self.file_view = MultipleFilesTreeWidget(compare_in_context_menu)
        self.file_view.header().close()
        self.save_state_btn = QPushButton("Save state")
        self.save_state_btn.setStyleSheet("QPushButton{font-weight: bold;}")
        self.load_files_btn = QPushButton("Load Files")
        self.load_recent_files_btn = QPushButton("Load recent Files")
        self.forget_btn = QPushButton("Forget")
        self.forget_all_btn = QPushButton("Forget all")

        self.save_state_btn.clicked.connect(self.save_state)
        self.forget_btn.clicked.connect(self.forget)
        self.load_files_btn.clicked.connect(self.load_files)
        self.load_recent_files_btn.clicked.connect(self.load_recent)
        self.file_view.itemDoubleClicked.connect(self.load_state)
        self.file_view.context_load.connect(self.load_state)
        self.forget_all_btn.clicked.connect(self.forget_all)
        self.last_point = None

        self.custom_names_chk = QCheckBox("Custom names")

        layout = QGridLayout()
        layout.addWidget(self.file_view, 0, 0, 1, 2)
        layout.addWidget(self.save_state_btn, 1, 0)
        layout.addWidget(self.load_files_btn, 1, 1)
        layout.addWidget(self.load_recent_files_btn, 2, 1)
        layout.addWidget(self.forget_btn, 2, 0)
        layout.addWidget(self.forget_all_btn, 3, 0)
        layout.addWidget(self.custom_names_chk, 3, 1)

        self.setLayout(layout)
        self.setMouseTracking(True)
        self.file_view.setMouseTracking(True)
        self.file_view.context_load.connect(self.load_state)
        self.file_view.context_compare.connect(self.load_compare)
        self.file_view.context_forget.connect(self.forget_action)
        self.error_list = []

        self._add_state.connect(self.save_state_action)
        self.settings.connect_("multiple_files_widget", self.view_changed)
        self.view_changed()

    def load_recent(self):
        dial = LoadRecentFiles(self.settings, self)
        if not dial.exec_():
            return

        dial_fun = ExecuteFunctionDialog(
            self.load_recent_fun, [dial.get_files()], exception_hook=load_data_exception_hook
        )
        dial_fun.exec_()

    def load_recent_fun(self, load_list, range_changed, step_changed):
        range_changed(0, len(load_list))
        for i, (file_list, method) in enumerate(load_list):
            load_class = self.load_register[method]
            state: ProjectInfoBase = load_class.load(file_list)
            self._add_state.emit(state, False)
            step_changed(i)
        for file_list, method in reversed(load_list):
            self.settings.add_last_files_multiple(file_list, method)

    def view_changed(self):
        self.setVisible(self.settings.get("multiple_files_widget", False))

    def execute_load_files(self, load_data: LoadProperty, range_changed, step_changed):
        range_changed(0, len(load_data.load_location))
        loaded_list = []
        for i, el in enumerate(load_data.load_location, 1):
            load_list = [el]
            while load_data.load_class.number_of_files() > len(load_list):
                load_list.append(load_data.load_class.get_next_file(load_list))
                if not os.path.exists(load_list[-1]):
                    self.error_list.append(el)
                    step_changed(i)
                    continue
            state: ProjectInfoBase = load_data.load_class.load(load_list)
            loaded_list.append((load_list, load_data.load_class.get_name()))
            self._add_state.emit(state, False)
            step_changed(i)

        for el in reversed(loaded_list):
            self.settings.add_last_files_multiple(*el)

    def load_files(self):
        dial = MultipleLoadDialog(self.load_register, self.settings.get_path_history())
        dial.setDirectory(self.settings.get("io.multiple_open_directory", str(Path.home())))
        dial.selectNameFilter(self.settings.get("io.multiple_open_filter", next(iter(self.load_register.keys()))))
        self.error_list = []
        if dial.exec_():
            result = dial.get_result()
            load_dir = os.path.dirname(result.load_location[0])
            self.settings.set("io.multiple_open_directory", load_dir)
            self.settings.add_path_history(load_dir)
            self.settings.set("io.multiple_open_filter", result.selected_filter)

            dial_fun = ExecuteFunctionDialog(self.execute_load_files, [result], exception_hook=load_data_exception_hook)
            dial_fun.exec_()
            if self.error_list:
                errors_message = QMessageBox()
                errors_message.setText("There are errors during load files")
                errors_message.setInformativeText("During load files cannot found some of files on disc")
                errors_message.setStandardButtons(QMessageBox.Ok)
                text = "\n".join(f"File: {x[0]}\n{x[1]}" for x in self.error_list)
                errors_message.setDetailedText(text)
                errors_message.exec_()

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
        # TODO left ellipsis
        if isinstance(state, list):
            self.add_states(state)
            return
        if not isinstance(state, ProjectInfoBase):  # workaround for PointsInfo load
            return
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
            clipped_text = metric.elidedText(normed_file_path, Qt.TextElideMode.ElideLeft, width)
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
        QTimer().singleShot(500, self.enable_forget)

    def forget_all(self):
        for index in range(self.file_view.topLevelItemCount(), -1, -1):
            self.forget_action(self.file_view.topLevelItem(index))

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

    @Slot()
    def enable_forget(self):
        self.forget_btn.setEnabled(True)

    def resizeEvent(self, event: QResizeEvent):
        metric = QFontMetrics(self.file_view.font())
        width = self.file_view.width() - 45
        for i, text in enumerate(self.file_list):
            clipped_text = metric.elidedText(text, Qt.TextElideMode.ElideLeft, width)
            item: QTreeWidgetItem = self.file_view.topLevelItem(i)
            item.setText(0, clipped_text)

    def mousePressEvent(self, event: QMouseEvent):
        if event.x() > self.width() - 20:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.x() > self.width() - 20:
            QApplication.setOverrideCursor(Qt.CursorShape.SplitHCursor)
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.ArrowCursor)
        if self.last_point is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        new_width = event.x() + 10
        new_width = max(new_width, 150)
        new_width = min(new_width, 600)

        self.setMinimumWidth(new_width)

    def leaveEvent(self, _):  # pylint: disable=no-self-use
        QApplication.setOverrideCursor(Qt.CursorShape.ArrowCursor)

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
        self.setFileMode(QFileDialog.FileMode.ExistingFiles)

    def accept(self):
        self.files_list.extend(self.selectedFiles())
        QFileDialog.accept(self)
