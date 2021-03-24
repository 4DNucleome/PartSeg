import os
import subprocess  # nosec
from sys import platform

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QGridLayout, QLabel, QPushButton


class DirectoryDialog(QDialog):
    def __init__(self, path_to_directory, additional_text=""):
        super().__init__()
        self.setWindowTitle("Path dialog")
        self.path_to_directory = path_to_directory
        text_label = QLabel(path_to_directory)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        open_btn = QPushButton("Open directory")
        open_btn.clicked.connect(self.open_folder)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout = QGridLayout()
        if additional_text != "":
            layout.addWidget(QLabel(additional_text + "<br><br>Path:"), 0, 0, 1, 2)
        else:
            layout.addWidget(QLabel("Path:"), 0, 0, 1, 2)
        layout.addWidget(text_label, 1, 0, 1, 2)
        layout.addWidget(close_btn, 2, 0)
        layout.addWidget(open_btn, 2, 1)
        self.setLayout(layout)

    def open_folder(self):
        if platform in ["linux", "linux2"]:
            subprocess.Popen(["xdg-open", self.path_to_directory])  # nosec
        elif platform == "darwin":
            subprocess.Popen(["open", self.path_to_directory])  # nosec
        elif platform == "win32":
            os.startfile(self.path_to_directory)  # nosec
        self.accept()
