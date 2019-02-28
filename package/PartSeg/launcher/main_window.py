import os
import importlib
from functools import partial
from qtpy.QtCore import QSize, Qt, QThread
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QMainWindow, QToolButton, QGridLayout, QWidget, QProgressBar, QMessageBox

from PartSeg.project_utils_qt.load_backup import import_config
from ..utils.global_settings import static_file_folder
from PartSeg.tiff_image import ImageReader


class Prepare(QThread):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.result = None
        self.errors = []

    def run(self):
        if self.module != "":
            from .. import plugins
            plugins.register()
            main_window_module = importlib.import_module(self.module)
            main_window = main_window_module.MainWindow
            settings = main_window.settings_class(main_window_module.CONFIG_FOLDER)
            self.errors = settings.load()
            reader = ImageReader()
            im = reader.read(main_window.initial_image_path)
            im.file_path = ""
            self.result = partial(main_window, settings=settings, initial_image=im)


class MainWindow(QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.lib_path = ""
        self.final_title = ""
        analysis_icon = QIcon(os.path.join(static_file_folder, 'icons', "icon.png"))
        stack_icon = QIcon(os.path.join(static_file_folder, 'icons', "icon_stack.png"))
        self.analysis_button = QToolButton(self)
        self.analysis_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.analysis_button.setIcon(analysis_icon)
        self.analysis_button.setText("Segmentation\nAnalysis")
        self.analysis_button.setIconSize(QSize(100, 100))
        self.mask_button = QToolButton(self)
        self.mask_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.mask_button.setIcon(stack_icon)
        self.mask_button.setText("Mask\nSegmentation")
        self.mask_button.setIconSize(QSize(100, 100))
        self.analysis_button.clicked.connect(self.launch_analysis)
        self.mask_button.clicked.connect(self.launch_mask)
        self.progress = QProgressBar()
        self.progress.setHidden(True)
        layout = QGridLayout()
        layout.addWidget(self.progress, 0, 0, 1, 2)
        layout.addWidget(self.analysis_button, 1, 0)
        layout.addWidget(self.mask_button, 1, 1)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowIcon(analysis_icon)
        self.prepare = None
        self.wind = None

    def launch_analysis(self):
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.analysis_button.setDisabled(True)
        self.mask_button.setDisabled(True)
        import_config()
        self.lib_path = "PartSeg.segmentation_analysis.main_window"
        self.final_title = "PartSeg Segmentation Analysis"
        self.prepare = Prepare(self.lib_path)
        self.prepare.finished.connect(self.launch)
        self.prepare.start()

    def launch_mask(self):
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.analysis_button.setDisabled(True)
        self.mask_button.setDisabled(True)
        import_config()
        self.lib_path = "PartSeg.segmentation_mask.stack_gui_main"
        self.final_title = "PartSeg Mask Segmentation"
        self.prepare = Prepare(self.lib_path)
        self.prepare.finished.connect(self.launch)
        self.prepare.start()

    def window_shown(self):
        self.close()

    def launch(self):
        if self.prepare.result is None:
            self.close()
            return
        if self.prepare.errors:
            errors_message = QMessageBox()
            errors_message.setText("There are errors during start")
            errors_message.setInformativeText("During load saved state some of data could not be load properly\n"
                                              "The files has prepared backup copies in  state directory (Help > State directory)")
            errors_message.setStandardButtons(QMessageBox.Ok)
            text = "\n".join(["File: " + x[0] + "\n" + str(x[1]) for x in self.prepare.errors])
            errors_message.setDetailedText(text)
            errors_message.exec()
        wind = self.prepare.result(title=self.final_title, signal_fun=self.window_shown)
        wind.show()
        self.wind = wind
