import os
import typing
from enum import Enum
from pathlib import Path

from qtpy.QtCore import QMimeData, QSize, Qt
from qtpy.QtGui import QDrag, QDragEnterEvent, QDragMoveEvent, QDropEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui.custom_load_dialog import CustomLoadDialog
from PartSeg.common_gui.stack_image_view import create_tool_button
from PartSeg.common_gui.universal_gui_part import EnumComboBox
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog
from PartSegCore.algorithm_describe_base import Register
from PartSegImage import Image


class ThickButton(QPushButton):
    def minimumWidth(self) -> int:
        return 20


class DragAndDropFileList(QListWidget):
    """from https://wiki.qt.io/QList_Drag_and_Drop_Example"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewMode(QListView.ListMode)
        self.setIconSize(QSize(55, 55))
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setDragEnabled(True)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setTextElideMode(Qt.ElideLeft)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def supportedDropActions(self):
        return Qt.MoveAction

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasFormat("application/x-item"):
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent):
        if event.mimeData().hasFormat("application/x-item"):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasFormat("application/x-item"):
            event.accept()
            event.setDropAction(Qt.MoveAction)

            name = event.mimeData().data("application/x-item").data().decode()
            item = QListWidgetItem(name)
            # item.setIcon(QIcon(":/images/iString")); // set path to image

            chose_item = self.itemAt(event.pos())
            if chose_item:
                index = self.row(chose_item)
                self.insertItem(index, item)
                self.setCurrentItem(item)
            else:
                self.addItem(item)
        else:
            event.ignore()

    def startDrag(self, supportedActions: typing.Union[Qt.DropActions, Qt.DropAction]) -> None:
        item = self.currentItem()
        mimeData = QMimeData()
        mimeData.setData("application/x-item", item.text().encode())
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        if drag.exec(Qt.MoveAction) == Qt.MoveAction:
            self.takeItem(self.row(item))


class FileList(QWidget):
    def __init__(self, read_method_dict: Register, settings: BaseSettings):
        super().__init__()
        self._file_list = DragAndDropFileList()
        self._file_list.setToolTip("Drag and drop to reorder files")
        self._add_files_btn = QPushButton("Add files")
        self._add_files_btn.clicked.connect(self._read_files)
        self._sort_btn = QPushButton("Sort")
        self._up_btn = create_tool_button("↑", None)
        self._up_btn.clicked.connect(self._move_up)
        self._down_btn = create_tool_button("↓", None)
        self._down_btn.clicked.connect(self._move_down)
        self._del_btn = create_tool_button("✕", None)
        self._del_btn.clicked.connect(self._del_entry)
        self._read_method_dict = read_method_dict
        self.settings = settings
        self._file_dict = {}
        layout = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout.addLayout(layout2)
        layout2.addWidget(self._file_list)
        layout3 = QVBoxLayout()
        layout2.addLayout(layout3)
        layout3.addStretch(1)
        layout3.addWidget(self._up_btn)
        layout3.addWidget(self._down_btn)
        layout3.addWidget(self._del_btn)
        layout3.addStretch(1)
        layout4 = QHBoxLayout()
        layout.addLayout(layout4)
        layout4.addWidget(self._add_files_btn)
        layout4.addWidget(self._sort_btn)
        self.setLayout(layout)

    def _move_up(self):
        index = self._file_list.currentIndex().row()
        if index < 1:
            return
        item = self._file_list.takeItem(index)
        self._file_list.insertItem(index - 1, item)
        self._file_list.setCurrentItem(item)

    def _move_down(self):
        index = self._file_list.currentIndex().row()
        if index in {-1, self._file_list.count() - 1}:
            return
        item = self._file_list.takeItem(index)
        self._file_list.insertItem(index + 1, item)
        self._file_list.setCurrentItem(item)

    def _del_entry(self):
        index = self._file_list.currentIndex().row()
        if index == -1:
            return
        self._file_list.takeItem(index)

    def _read_files(self):
        load_dict = {x: y for x, y in self._read_method_dict.items() if y.number_of_files() == 1 and not y.partial()}
        dial = CustomLoadDialog(load_dict, history=self.settings.get_path_history())
        dial.setFileMode(CustomLoadDialog.ExistingFiles)
        dial.setDirectory(self.settings.get("io.open_directory", str(Path.home())))
        dial.selectNameFilter(self.settings.get("io.open_filter", next(iter(load_dict))))
        if dial.exec_():
            result = dial.get_result()
            self.settings.set("io.open_filter", result.selected_filter)
            load_dir = os.path.dirname(result.load_location[0])
            self.settings.set("io.open_directory", load_dir)
            self.settings.add_path_history(load_dir)

            def read_multiple(range_changed, step_changed):
                range_changed(0, len(result.load_location))
                res = []
                for i, el in enumerate(result.load_location, 1):
                    res.append(result.load_class.load([el], metadata={"default_spacing": self.settings.image_spacing}))
                    step_changed(i)
                return res

            dial2 = ExecuteFunctionDialog(read_multiple)
            if dial2.exec():
                result = dial2.get_result()
                self.add_files(result)

    def add_file(self, image: Image) -> None:
        if image.file_path in self._file_dict:
            return
        self._file_dict[image.file_path] = image
        self._file_list.addItem(image.file_path)

    def add_files(self, images: typing.List[Image]) -> None:
        for image in images:
            if image.file_path in self._file_dict:
                continue
            self._file_dict[image.file_path] = image
            self._file_list.addItem(image.file_path)


class ImportDialog(QDialog):
    def __init__(self, load_dict: Register, settings: BaseSettings):
        super().__init__()
        self.settings = settings
        self.load_dict = load_dict
        self.file_list_groups: typing.List[typing.Tuple[EnumComboBox, FileList]] = []
        self.new_group_btn = QPushButton("New group")
        layout = QVBoxLayout()
        self.layout_files = QHBoxLayout()
        layout.addLayout(self.layout_files)
        layout.addWidget(self.new_group_btn)
        self.setLayout(layout)
        self.add_group()

    def add_group(self):
        choose_dim = EnumComboBox(DimEnum)
        file_list = FileList(self.load_dict, self.settings)
        lay = QVBoxLayout()
        lay.addWidget(choose_dim)
        lay.addWidget(file_list)
        self.layout_files.addLayout(lay)
        self.file_list_groups.append((choose_dim, file_list))


class DimEnum(Enum):
    Channel = "C"
    Stack = "Z"
    Time = "T"

    def __str__(self):
        return self.name
