from glob import glob
from global_settings import use_qt5


if use_qt5:
    raise NotImplementedError("Pyqt5 support not implemented")
    pass
else:
    from PyQt4.QtCore import Qt, QTimer
    from PyQt4.QtGui import QWidget, QLineEdit, QPushButton, QDockWidget, QDialog, QListWidget, QAbstractItemView, \
        QVBoxLayout, QHBoxLayout, QMessageBox

__author__ = "Grzegorz Bokota"


class AcceptFiles(QDialog):
    def __init__(self, files):
        super(AcceptFiles, self).__init__()
        self.ok = QPushButton("Add", self)
        self.ok.pyqtConfigure(clicked=self.accept)
        discard = QPushButton("Discard", self)
        discard.pyqtConfigure(clicked=self.close)
        self.files = QListWidget(self)
        self.files.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for file_name in files:
            self.files.addItem(file_name)
        for i in range(self.files.count()):
            self.files.item(i).setSelected(True)
        self.ok.setDefault(True)
        self.ok.setAutoDefault(True)

        layout = QVBoxLayout()
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

    def __init__(self, parent):
        """TODO: to be defined1. """
        QWidget.__init__(self, parent)
        self.files_to_proceed = set()
        self.paths = QLineEdit(self)
        self.selected_files = QListWidget(self)
        self.found_button = QPushButton("Find all", self)
        self.open_button = QPushButton("Select")

    def find_all(self):
        paths = glob(str(self.paths.text()))
        paths = sorted(list(set(paths) - self.files_to_proceed))
        if len(paths) > 0:
            dialog = AcceptFiles(paths)
            if dialog.exec_():
                new_paths = dialog.get_files()
                self.selected_files.addItems(new_paths)
        else:
            QMessageBox.warning(self, "No new files", "No new files found", QMessageBox.Ok)


class BatchWindow(QDockWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.files_to_proceed = set()


