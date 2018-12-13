import os

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QToolButton, QHBoxLayout, QWidget
from partseg_utils.global_settings import static_file_folder


class MainWindow(QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        analysis_icon = QIcon(os.path.join(static_file_folder, 'icons', "icon.png"))
        stack_icon = QIcon(os.path.join(static_file_folder, 'icons', "icon_stack.png"))
        self.partseg_button = QToolButton(self)
        self.partseg_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.partseg_button.setIcon(analysis_icon)
        self.partseg_button.setText("Segmentation\nAnalysis")
        self.partseg_button.setIconSize(QSize(100, 100))
        self.stackseg_button = QToolButton(self)
        self.stackseg_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.stackseg_button.setIcon(stack_icon)
        self.stackseg_button.setText("Mask\nSegmentation")
        self.stackseg_button.setIconSize(QSize(100, 100))
        self.partseg_button.clicked.connect(self.launch_partseg)
        self.stackseg_button.clicked.connect(self.launch_stackseg)
        layout = QHBoxLayout()
        layout.addWidget(self.partseg_button)
        layout.addWidget(self.stackseg_button)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowIcon(analysis_icon)

    def launch_partseg(self):
        from partseg2.main_window import MainWindow
        self.launch(MainWindow, "PartSeg Segmentation Analysis")

    def launch_stackseg(self):
        from stackseg.stack_gui_main import MainWindow
        self.launch(MainWindow, "PartSeg Mask Segmentation")

    def window_shown(self):
        self.close()

    def launch(self, cls, title):
        print(self.parent())
        wind = cls(title, self.window_shown)
        wind.show()
        self.wind = wind
