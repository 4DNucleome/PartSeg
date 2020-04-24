import typing

from qtpy.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QDrag
from qtpy.QtCore import Qt, QSize, QMimeData
from qtpy.QtWidgets import (
    QDialog,
    QPushButton,
    QWidget,
    QListWidget,
    QHBoxLayout,
    QAbstractItemView,
    QListView,
    QListWidgetItem,
    QVBoxLayout,
)

from PartSeg.common_gui.stack_image_view import create_tool_button


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
    def __init__(self):
        super().__init__()
        self._file_list = DragAndDropFileList()
        self._file_list.setToolTip("Drag and drop to reorder files")
        self._add_files_btn = QPushButton("Add files")
        self._sort_btn = QPushButton("Sort")
        self._up_btn = create_tool_button("↑", None)
        self._up_btn.clicked.connect(self._move_up)
        self._down_btn = create_tool_button("↓", None)
        self._down_btn.clicked.connect(self._move_down)
        self._del_btn = create_tool_button("✕", None)
        self._del_btn.clicked.connect(self._del_entry)
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

    def add_file(self, path: str) -> None:
        self._file_list.addItem(path)

    def add_files(self, paths: typing.List[str]) -> None:
        self._file_list.addItems(paths)


class ImportDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.new_group_btn = QPushButton("Add file ")
