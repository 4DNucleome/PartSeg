import os
import tifffile as tif
from qt_import import QMainWindow, QPixmap, QImage, QPushButton, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from stack_settings import Settings

import matplotlib
from matplotlib import colors
import numpy as np


class MainMenu(QWidget):
    def __init__(self, settings):
        super(MainMenu, self).__init__()
        self.settings = settings
        self.load_btn = QPushButton("Load")
        self.save_btn = QPushButton("Save")
        layout = QHBoxLayout()
        layout.addWidget(self.load_btn)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

    def load_image(self):
        dial = QFileDialog()
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setDirectory(self.settings.open_directory)
        filters = ["raw image (*.tiff *.tif *.lsm)"]
        dial.setNameFilters(filters)
        if not dial.exec_():
            return
        file_path = str(dial.selectedFiles()[0])
        self.settings.open_directory = os.path.dirname(str(file_path))
        im = tif.imread(file_path)
        self.settings.image = im


class MainWindow(QMainWindow):
    def __init__(self, title):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.settings = Settings()
        self.main_menu = MainMenu(self.settings)

        self.pixmap = QLabel()

        im = tif.imread("/home/czaki/Obrazy/160112_Chr1aqua_q488_pred_TOPRO/B/B.lsm")[0, 20, 3]
        width, height = im.shape
        im = colors.PowerNorm(gamma=1, vmin=im.min(), vmax=im.max())(im)
        cmap = matplotlib.cm.get_cmap("gray")
        colored_image = cmap(im)
        # noinspection PyTypeChecker
        im = np.array(colored_image * 255, dtype=np.uint8)
        print(im.shape)
        print(im.max(axis=2))
        print(im.min(axis=2))
        im2 = QImage(im.data, width, height, im.dtype.itemsize*width*4, QImage.Format_ARGB32)

        self.pixmap.setPixmap(QPixmap.fromImage(im2))

        layout = QVBoxLayout()
        layout.addWidget(self.main_menu)
        layout.addWidget(self.pixmap)
        self.widget = QWidget()
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)



