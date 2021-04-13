import os
from functools import partial
from glob import glob
from pathlib import Path
from typing import List

from qtpy.QtCore import QPoint, Qt, Signal
from qtpy.QtGui import QDragEnterEvent, QDropEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PartSegCore.analysis.calculation_plan import MaskMapper

from ..common_backend.base_settings import BaseSettings


class AcceptFiles(QDialog):
    def __init__(self, files):
        super().__init__()
        self.ok = QPushButton("Add", self)
        self.ok.clicked.connect(self.accept)
        discard = QPushButton("Discard", self)
        discard.clicked.connect(self.close)
        self.files = QListWidget(self)
        self.files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for file_name in files:
            self.files.addItem(file_name)
        for i in range(self.files.count()):
            self.files.item(i).setSelected(True)
        self.ok.setDefault(True)
        self.ok.setAutoDefault(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Found {len(files)} files"))
        layout.addWidget(self.files)
        butt_layout = QHBoxLayout()
        butt_layout.addWidget(discard)
        butt_layout.addStretch()
        butt_layout.addWidget(self.ok)
        layout.addLayout(butt_layout)
        self.setLayout(layout)

    def selection_changed(self):
        if self.files.selectedItems().count() == 0:
            self.ok.setDisabled(True)
        else:
            self.ok.setEnabled(True)

    def get_files(self):
        return [str(item.text()) for item in self.files.selectedItems()]


class FileListItem(QListWidgetItem):
    def __init__(self, file_path):
        size = os.stat(file_path).st_size
        size = float(size) / (1024 ** 2)
        super().__init__(f"{file_path:s} ({size:.2f} MB)")
        self.setTextAlignment(Qt.AlignRight)
        self.file_path = file_path


class FileListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)


class AddFiles(QWidget):
    """Docstring for AddFiles. """

    file_list_changed = Signal(set)

    def __init__(self, settings: BaseSettings, parent=None, btn_layout=QHBoxLayout):
        """TODO: to be defined1. """
        QWidget.__init__(self, parent)
        self.mask_list: List[MaskMapper] = []
        self.settings = settings
        self.files_to_proceed = set()
        self.paths_input = QLineEdit(self)
        self.selected_files = FileListWidget(self)
        self.selected_files.itemSelectionChanged.connect(self.file_chosen)
        self.found_button = QPushButton("Find all", self)
        self.found_button.clicked.connect(self.find_all)
        self.select_files_button = QPushButton("Select files")
        self.select_dir_button = QPushButton("Select directory")
        self.select_files_button.clicked.connect(self.select_files)
        self.select_dir_button.clicked.connect(self.select_directory)
        self.delete_button = QPushButton("Remove file", self)
        self.delete_button.setDisabled(True)
        self.delete_button.clicked.connect(self.delete_element)
        self.clean_button = QPushButton("Remove all", self)
        self.clean_button.clicked.connect(self.clean)
        layout = QVBoxLayout()
        layout.addWidget(self.paths_input)
        select_layout = btn_layout()
        select_layout.addWidget(self.select_files_button)
        select_layout.addWidget(self.select_dir_button)
        select_layout.addWidget(self.found_button)
        select_layout.addStretch()
        select_layout.addWidget(self.clean_button)
        select_layout.addWidget(self.delete_button)
        layout.addLayout(select_layout)
        layout.addWidget(self.selected_files)
        self.setLayout(layout)
        self.setAcceptDrops(True)
        self.selected_files.customContextMenuRequested.connect(self.files_context_menu)

    def files_context_menu(self, point: QPoint):
        element = self.selected_files.itemAt(point)
        if element is None:
            return
        menu = QMenu()
        menu.addAction("Load image").triggered.connect(self._load_file)
        for mask_def in self.mask_list:
            menu.addAction(f"Load with mask '{mask_def.name}'").triggered.connect(
                partial(self._load_file_with_mask, mask_def)
            )
        menu.addAction("Delete").triggered.connect(self.delete_element)
        menu.exec_(self.selected_files.mapToGlobal(point))

    def _load_file(self):
        file_path = self.selected_files.item(self.selected_files.currentRow()).file_path
        self.settings._load_files_call([file_path])  # pylint: disable=W0212

    def _load_file_with_mask(self, mask_mapper: MaskMapper):
        file_path = self.selected_files.item(self.selected_files.currentRow()).file_path
        mask_path = mask_mapper.get_mask_path(file_path)
        self.settings._load_files_call([file_path, mask_path])  # pylint: disable=W0212

    def dragEnterEvent(self, event: QDragEnterEvent):  # pylint: disable=R0201
        if event.mimeData().hasFormat("text/plain"):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files_list = event.mimeData().text().split()
        self.parse_drop_file_list(files_list)

    def parse_drop_file_list(self, files_list):
        res_list = []
        base_path = self.paths_input.text()
        for file_path in files_list:
            if os.path.isabs(file_path):
                res_list.append(file_path)
            else:
                res_list.append(os.path.join(base_path, file_path))
        missed_files = [x for x in res_list if not os.path.exists(x)]
        if missed_files:
            if len(missed_files) > 6:
                missed_files = missed_files[:6] + ["..."]
            missed_files_str = "<br>".join(missed_files)
            QMessageBox().warning(
                self,
                "Missed Files",
                f"Cannot find files:<br>{missed_files_str}<br>Set proper base directory using <i>Select directory</i>",
            )
        else:
            self.update_files_list(res_list)

    def find_all(self):
        paths = glob(str(self.paths_input.text()))
        paths = sorted([x for x in (set(paths) - self.files_to_proceed) if not os.path.isdir(x)])
        if len(paths) > 0:
            self.update_files_list(paths)

        else:
            QMessageBox.warning(self, "No new files", "No new files found", QMessageBox.Ok)

    def update_files_list(self, paths):
        dialog = AcceptFiles(paths)
        if dialog.exec_():
            new_paths = dialog.get_files()
            for path in new_paths:
                self.selected_files.addItem(FileListItem(path))
            self.files_to_proceed.update(new_paths)
            self.file_list_changed.emit(self.files_to_proceed)

    def select_files(self):
        dial = QFileDialog(self, "Select files")
        dial.setDirectory(
            self.settings.get("io.batch_directory", self.settings.get("io.load_image_directory", str(Path.home())))
        )
        dial.setFileMode(QFileDialog.ExistingFiles)
        if dial.exec_():
            self.settings.set("io.batch_directory", os.path.dirname(str(dial.selectedFiles()[0])))
            new_paths = sorted(set(map(str, dial.selectedFiles())) - self.files_to_proceed)
            for path in new_paths:
                self.selected_files.addItem(FileListItem(path))
            self.files_to_proceed.update(new_paths)
            self.file_list_changed.emit(self.files_to_proceed)

    def select_directory(self):
        dial = QFileDialog(self, "Select directory")
        dial.setDirectory(
            self.settings.get("io.batch_directory", self.settings.get("io.load_image_directory", str(Path.home())))
        )
        dial.setFileMode(QFileDialog.Directory)
        if dial.exec_():
            self.paths_input.setText(dial.selectedFiles()[0])
            self.settings.set("io.batch_directory", str(dial.selectedFiles()[0]))

    def file_chosen(self):
        self.delete_button.setEnabled(True)

    def delete_element(self):
        item = self.selected_files.takeItem(self.selected_files.currentRow())
        self.files_to_proceed.remove(item.file_path)
        self.file_list_changed.emit(self.files_to_proceed)
        if self.selected_files.count() == 0:
            self.delete_button.setDisabled(True)

    def clean(self):
        self.selected_files.clear()
        self.files_to_proceed.clear()
        self.file_list_changed.emit(self.files_to_proceed)

    def get_paths(self):
        return list(sorted(self.files_to_proceed))
