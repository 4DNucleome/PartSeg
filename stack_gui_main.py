from __future__ import division
import tifffile as tif
from qt_import import QMainWindow, QPixmap, QImage, QPushButton, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, \
    QLabel, QScrollArea, QPalette, QSizePolicy, QToolButton, QIcon, QSize, QAction, Qt, QPainter, QPen, \
    QColor, QScrollBar, QApplication, pyqtSignal, QPoint, QSlider, QSpinBox, QComboBox, QTabWidget, QDoubleSpinBox, \
    QFormLayout, QAbstractSpinBox, QStackedLayout
from stack_settings import ImageSettings
from stack_image_view import ImageView
from universal_gui_part import right_label, Spacing
from universal_const import UNITS_LIST
from stack_algorithm.algorithm_description import stack_algorithm_dict, AlgorithmSettingsWidget


import matplotlib
from matplotlib import colors
import numpy as np
import os


class MainMenu(QWidget):
    image_loaded = pyqtSignal()

    def __init__(self, settings):
        super(MainMenu, self).__init__()
        self.settings = settings
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_image)
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
        self.image_loaded.emit()


class AlgorithmOptions(QWidget):
    def __init__(self, settings):
        super(AlgorithmOptions, self).__init__()
        self.algorithm_choose = QComboBox()
        self.execute_btn = QPushButton("Execute")
        self.stack_layout = QStackedLayout()
        for name, val in stack_algorithm_dict.items():
            self.algorithm_choose.addItem(name)
            widget = AlgorithmSettingsWidget(val, settings)
            self.stack_layout.addWidget(widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.execute_btn)
        main_layout.addWidget(self.algorithm_choose)
        main_layout.addLayout(self.stack_layout)
        main_layout.addStretch()
        self.setLayout(main_layout)

        self.algorithm_choose.currentIndexChanged.connect(self.stack_layout.setCurrentIndex)
        self.execute_btn.clicked.connect(self.execute_action)

    def execute_action(self):
        widget = self.stack_layout.currentWidget()
        print(widget.get_values())


class ImageInformation(QWidget):
    def __init__(self, settings, parent=None):
        super(ImageInformation, self).__init__(parent)
        self._settings = settings
        self.path = QLabel("<b>Path:</b>")
        self.spacing = [QDoubleSpinBox() for _ in range(3)]
        for el in self.spacing:
            el.setAlignment(Qt.AlignRight)
            el.setButtonSymbols(QAbstractSpinBox.NoButtons)
            el.setRange(0, 1000)
        self.units = QComboBox()
        self.units.addItems(UNITS_LIST)

        spacing_layout = QFormLayout()
        spacing_layout.addRow("x:", self.spacing[0])
        spacing_layout.addRow("y:", self.spacing[1])
        spacing_layout.addRow("z:", self.spacing[2])
        spacing_layout.addRow("Units:", self.units)

        layout = QVBoxLayout()
        layout.addWidget(self.path)
        layout.addWidget(QLabel("Image spacing:"))
        layout.addLayout(spacing_layout)
        layout.addStretch()
        self.setLayout(layout)


class Options(QTabWidget):
    def __init__(self, settings, parent=None):
        super(Options, self).__init__(parent)
        self._settings = settings
        self.algorithm_options = AlgorithmOptions(settings)
        self.image_properties = ImageInformation(settings, parent)
        self.addTab(self.image_properties, "Image")
        self.addTab(self.algorithm_options, "Segmentation")


class MainWindow(QMainWindow):
    def __init__(self, title):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.settings = ImageSettings()
        self.main_menu = MainMenu(self.settings)
        self.image_view = ImageView()
        self.algorithm_options = Options(self.settings)
        self.main_menu.image_loaded.connect(self.image_read)

        # self.scroll_area.setVisible(False)
        # self.scroll_area.setS

        im = tif.imread("stack.tif")
        # width, height = im.shape
        # im = colors.PowerNorm(gamma=1, vmin=im.min(), vmax=im.max())(im)
        # cmap = matplotlib.cm.get_cmap("cubehelix")
        # colored_image = cmap(im)
        # noinspection PyTypeChecker
        # im = np.array(colored_image * 255, dtype=np.uint8)
        # im2 = QImage(im.data, width, height, im.dtype.itemsize*width*4, QImage.Format_ARGB32)

        # self.pixmap.setPixmap(QPixmap.fromImage(im2))
        # self.im_view = pg.ImageView(self)
        # self.im_view.setImage(im)
        layout = QVBoxLayout()
        layout.addWidget(self.main_menu)
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.image_view, 1)
        sub_layout.addWidget(self.algorithm_options, 0)
        layout.addLayout(sub_layout)
        # self.pixmap.adjustSize()
        # self.pixmap.update_size(2)
        self.widget = QWidget()
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)
        self.image_view.set_image(im)

    def image_read(self):
        print("buka1", self.settings.image.shape)
        self.image_view.set_image(self.settings.image)




