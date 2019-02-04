from glob import glob
import os
from pathlib import Path

from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QListWidget, QLineEdit, QListWidgetItem, \
    QMessageBox, QDialog, QAbstractItemView, QLabel, QFileDialog
from ..project_utils_qt.settings import BaseSettings


class AcceptFiles(QDialog):
    def __init__(self, files):
        super(AcceptFiles, self).__init__()
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
        layout.addWidget(QLabel("Found {} files".format(len(files))))
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


class AddFiles(QWidget):
    """Docstring for AddFiles. """

    file_list_changed = Signal(set)

    def __init__(self, settings: BaseSettings, parent=None, btn_layout=QHBoxLayout):
        """TODO: to be defined1. """
        QWidget.__init__(self, parent)
        self.settings = settings
        self.files_to_proceed = set()
        self.paths = QLineEdit(self)
        self.selected_files = QListWidget(self)
        self.selected_files.itemSelectionChanged.connect(self.file_chosen)
        self.found_button = QPushButton("Find all", self)
        self.found_button.clicked.connect(self.find_all)
        self.select_files_button = QPushButton("Select files")
        self.select_dir_button = QPushButton("Select directory")
        self.select_files_button.clicked.connect(self.select_files)
        self.select_dir_button.clicked.connect(self.select_directory)
        self.delete_button = QPushButton("Remove file from list", self)
        self.delete_button.setDisabled(True)
        self.delete_button.clicked.connect(self.delete_element)
        self.clean_button = QPushButton("Clean file List", self)
        self.clean_button.clicked.connect(self.clean)
        layout = QVBoxLayout()
        layout.addWidget(self.paths)
        select_layout = btn_layout()
        select_layout.addWidget(self.found_button)
        select_layout.addWidget(self.select_files_button)
        select_layout.addWidget(self.select_dir_button)
        select_layout.addStretch()
        select_layout.addWidget(self.clean_button)
        select_layout.addWidget(self.delete_button)
        layout.addLayout(select_layout)
        layout.addWidget(self.selected_files)
        self.setLayout(layout)

    def find_all(self):
        paths = glob(str(self.paths.text()))
        paths = sorted([x for x in (set(paths) - self.files_to_proceed) if not os.path.isdir(x)])
        if len(paths) > 0:
            dialog = AcceptFiles(paths)
            if dialog.exec_():
                new_paths = dialog.get_files()
                for path in new_paths:
                    size = os.stat(path).st_size
                    size = float(size) / (1024 ** 2)
                    lwi = QListWidgetItem("{:s} ({:.2f} MB)".format(path, size))
                    lwi.setTextAlignment(Qt.AlignRight)
                    self.selected_files.addItem(lwi)
                self.files_to_proceed.update(new_paths)
                self.file_list_changed.emit(self.files_to_proceed)
        else:
            QMessageBox.warning(self, "No new files", "No new files found", QMessageBox.Ok)

    def select_files(self):
        dial = QFileDialog(self, "Select files")
        dial.setDirectory(
            self.settings.get("io.batch_directory", self.settings.get("io.load_image_directory", str(Path.home()))))
        dial.setFileMode(QFileDialog.ExistingFiles)
        if dial.exec_():
            self.settings.set("io.batch_directory", os.path.dirname(str(dial.selectedFiles()[0])))
            new_paths = sorted(set(map(str, dial.selectedFiles())) - self.files_to_proceed)
            for path in new_paths:
                size = os.stat(path).st_size
                size = float(size) / (1024 ** 2)
                lwi = QListWidgetItem("{:s} ({:.2f} MB)".format(path, size))
                lwi.setTextAlignment(Qt.AlignRight)
                self.selected_files.addItem(lwi)
            self.files_to_proceed.update(new_paths)
            self.file_list_changed.emit(self.files_to_proceed)

    def select_directory(self):
        dial = QFileDialog(self, "Select directory")
        dial.setDirectory(
            self.settings.get("io.batch_directory", self.settings.get("io.load_image_directory", str(Path.home()))))
        dial.setFileMode(QFileDialog.Directory)
        if dial.exec_():
            self.paths.setText(dial.selectedFiles()[0])
            self.settings.set("io.batch_directory", str(dial.selectedFiles()[0]))

    def file_chosen(self):
        self.delete_button.setEnabled(True)

    def delete_element(self):
        item = self.selected_files.takeItem(self.selected_files.currentRow())
        path = str(item.text())
        path = path[:path.rfind("(") - 1]
        self.files_to_proceed.remove(path)
        self.file_list_changed.emit(self.files_to_proceed)
        if self.selected_files.count() == 0:
            self.delete_button.setDisabled(True)

    def clean(self):
        self.selected_files.clear()
        self.files_to_proceed.clear()
        self.file_list_changed.emit(self.files_to_proceed)

    def get_paths(self):
        return list(sorted(self.files_to_proceed))
