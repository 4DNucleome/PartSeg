import importlib
import os
from functools import partial

from qtpy.QtCore import QSize, Qt, QThread
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QGridLayout, QMainWindow, QMessageBox, QProgressBar, QToolButton, QWidget

from PartSeg import ANALYSIS_NAME, APP_NAME, MASK_NAME
from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_backend.load_backup import import_config
from PartSeg.common_gui.main_window import BaseMainWindow
from PartSegData import icons_dir
from PartSegImage import TiffImageReader


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
            main_window: BaseMainWindow = main_window_module.MainWindow
            settings: BaseSettings = main_window.get_setting_class()(main_window_module.CONFIG_FOLDER)
            self.errors = settings.load()
            reader = TiffImageReader()
            im = reader.read(main_window.initial_image_path)
            im.file_path = ""
            self.result = partial(main_window, settings=settings, initial_image=im)


class MainWindow(QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.lib_path = ""
        self.final_title = ""
        analysis_icon = QIcon(os.path.join(icons_dir, "icon.png"))
        stack_icon = QIcon(os.path.join(icons_dir, "icon_stack.png"))
        self.analysis_button = QToolButton(self)
        self.analysis_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.analysis_button.setIcon(analysis_icon)
        # TODO use more general solution for text wrapping
        self.analysis_button.setText(ANALYSIS_NAME.replace(" ", "\n"))
        self.analysis_button.setIconSize(QSize(100, 100))
        self.mask_button = QToolButton(self)
        self.mask_button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.mask_button.setIcon(stack_icon)
        self.mask_button.setText(MASK_NAME.replace(" ", "\n"))
        self.mask_button.setIconSize(QSize(100, 100))
        self.analysis_button.clicked.connect(self.launch_analysis)
        self.mask_button.clicked.connect(self.launch_mask)
        self.progress = QProgressBar()
        self.progress.setHidden(True)
        layout = QGridLayout()
        layout.addWidget(self.progress, 0, 0, 1, 2)
        layout.addWidget(self.analysis_button, 1, 1)
        layout.addWidget(self.mask_button, 1, 0)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.setWindowIcon(analysis_icon)
        self.prepare = None
        self.wind = None

    def _launch_begin(self):
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.analysis_button.setDisabled(True)
        self.mask_button.setDisabled(True)
        import_config()

    def launch_analysis(self):
        self._launch_begin()
        self._launch_analysis()
        self.prepare.start()

    def _launch_analysis(self):
        self.lib_path = "PartSeg._roi_analysis.main_window"
        self.final_title = f"{APP_NAME} {ANALYSIS_NAME}"
        self.prepare = Prepare(self.lib_path)
        self.prepare.finished.connect(self.launch)

    def launch_mask(self):
        self._launch_begin()
        self._launch_mask()
        self.prepare.start()

    def _launch_mask(self):
        self.lib_path = "PartSeg._roi_mask.main_window"
        self.final_title = f"{APP_NAME} {MASK_NAME}"
        self.prepare = Prepare(self.lib_path)
        self.prepare.finished.connect(self.launch)

    def window_shown(self):
        self.close()

    def launch(self):
        if self.prepare.result is None:
            self.close()
            return
        if self.prepare.errors:
            errors_message = QMessageBox()
            errors_message.setText("There are errors during start")
            errors_message.setInformativeText(
                "During load saved state some of data could not be load properly\n"
                "The files has prepared backup copies in  state directory (Help > State directory)"
            )
            errors_message.setStandardButtons(QMessageBox.Ok)
            text = "\n".join("File: " + x[0] + "\n" + str(x[1]) for x in self.prepare.errors)

            errors_message.setDetailedText(text)
            errors_message.exec()
        wind = self.prepare.result(title=self.final_title, signal_fun=self.window_shown)
        wind.show()
        self.wind = wind
