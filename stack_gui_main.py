from __future__ import division
import tifffile as tif
from qt_import import QMainWindow, QPixmap, QImage, QPushButton, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, \
    QLabel, QScrollArea, QPalette, QSizePolicy, QToolButton, QIcon, QSize, QAction, Qt, QPainter, QPen, \
    QColor, QScrollBar, QApplication, pyqtSignal, QPoint, QSlider, QSpinBox, QComboBox, QTabWidget, QDoubleSpinBox, \
    QFormLayout, QAbstractSpinBox, QStackedLayout, QCheckBox
from stack_settings import ImageSettings
from stack_image_view import ImageView
from universal_gui_part import right_label, Spacing
from universal_const import UNITS_LIST
from stack_algorithm.algorithm_description import stack_algorithm_dict, AlgorithmSettingsWidget
from flow_layout import FlowLayout

import matplotlib
from matplotlib import colors
import numpy as np
import os

from batch_window import AddFiles


class MainMenu(QWidget):
    image_loaded = pyqtSignal()

    def __init__(self, settings):
        """
        :type settings: ImageSettings
        :param settings:
        """
        super(MainMenu, self).__init__()
        self.settings = settings
        self.load_image_btn = QPushButton("Load image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_segmentation_btn = QPushButton("Load segmentation")

        self.save_segmentation_btn = QPushButton("Save segmentation")
        self.save_cutted_parts = QPushButton("Save results")
        layout = QHBoxLayout()
        layout.addWidget(self.load_image_btn)
        layout.addWidget(self.load_segmentation_btn)
        layout.addWidget(self.save_cutted_parts)
        layout.addWidget(self.save_segmentation_btn)
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
        self.settings.image = im, file_path
        # self.image_loaded.emit()


class ChosenComponents(QWidget):
    def __init__(self):
        super(ChosenComponents, self).__init__()
        self.setLayout(FlowLayout())
        self.check_box = dict()

    def set_chose(self, components_index, chosen_components):
        widget = QWidget()
        widget.setLayout(self.layout())
        self.setLayout(FlowLayout())
        self.check_box.clear()
        chosen_components = set(chosen_components)
        for el in components_index:
            check = QCheckBox(str(el))
            if el in chosen_components:
                check.setChecked(True)
            self.check_box[el] = check
            self.layout().addWidget(check)
        self.update()

    def change_state(self, num, val):
        self.check_box[num].setChecked(val)

    def get_chosen(self):
        res = []
        for num, check in self.check_box.items():
            if check.isChecked():
                res.append(num)
        return res


class AlgorithmOptions(QWidget):
    labels_changed = pyqtSignal(np.ndarray)

    def __init__(self, settings, control_view):
        """
        :type control_view: ImageState
        :type settings: ImageSettings
        :param settings:
        :param control_view:
        """
        super(AlgorithmOptions, self).__init__()
        self.old_segmentation = None
        self.algorithm_choose = QComboBox()
        self.show_result = QCheckBox("Show result")
        self.show_result.setChecked(control_view.show_label)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0, 1)
        self.opacity.setSingleStep(0.1)
        self.opacity.setValue(control_view.opacity)
        self.only_borders = QCheckBox("Only borders")
        self.only_borders.setChecked(control_view.only_borders)
        self.borders_thick = QSpinBox()
        self.borders_thick.setRange(1, 10)
        self.borders_thick.setSingleStep(1)
        self.borders_thick.setValue(control_view.borders_thick)
        self.execute_btn = QPushButton("Execute")
        self.execute_all_btn = QPushButton("Execute all")
        self.stack_layout = QStackedLayout()
        self.choose_components = ChosenComponents()
        for name, val in stack_algorithm_dict.items():
            self.algorithm_choose.addItem(name)
            widget = AlgorithmSettingsWidget(settings, *val)
            self.stack_layout.addWidget(widget)

        main_layout = QVBoxLayout()
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(self.show_result)
        opt_layout.addWidget(right_label("Opacity:"))
        opt_layout.addWidget(self.opacity)
        main_layout.addLayout(opt_layout)
        opt_layout2 = QHBoxLayout()
        opt_layout2.addWidget(self.only_borders)
        opt_layout2.addWidget(right_label("Border thick:"))
        opt_layout2.addWidget(self.borders_thick)
        main_layout.addLayout(opt_layout2)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.execute_btn)
        btn_layout.addWidget(self.execute_all_btn)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.algorithm_choose)
        main_layout.addLayout(self.stack_layout)
        main_layout.addWidget(self.choose_components)
        main_layout.addStretch()
        self.setLayout(main_layout)

        self.algorithm_choose.currentIndexChanged.connect(self.stack_layout.setCurrentIndex)
        self.execute_btn.clicked.connect(self.execute_action)
        self.opacity.valueChanged.connect(control_view.set_opacity)
        self.show_result.stateChanged.connect(control_view.set_show_label)
        self.only_borders.stateChanged.connect(control_view.set_borders)
        self.borders_thick.valueChanged.connect(control_view.set_borders_thick)
        settings.image_changed.connect(self.image_changed)

    def image_changed(self):
        print("buka2")
        self.old_segmentation = None
        self.choose_components.set_chose([], [])

    def execute_action(self):
        self.execute_btn.setDisabled(True)
        chosen = sorted(self.choose_components.get_chosen())
        if len(chosen) == 0:
            blank = None
        else:
            if len(chosen) > 250:
                blank = np.zeros(self.old_segmentation.shape, dtype=np.uint16)
            else:
                blank = np.zeros(self.old_segmentation.shape, dtype=np.uint8)
            for i, v in enumerate(chosen):
                blank[self.old_segmentation == v] = i+1

        widget = self.stack_layout.currentWidget()
        segmentation = widget.execute(blank)
        self.old_segmentation = segmentation
        self.choose_components.set_chose(range(1, segmentation.max()+1), np.arange(len(chosen))+1)
        self.execute_btn.setEnabled(True)
        self.labels_changed.emit(segmentation)


class ImageInformation(QWidget):
    def __init__(self, settings, parent=None):
        """:type settings: ImageSettings"""
        super(ImageInformation, self).__init__(parent)
        self._settings = settings
        self.path = QLabel("<b>Path:</b> example image")
        self.path.setWordWrap(True)
        self.spacing = [QDoubleSpinBox() for _ in range(3)]
        for i, el in enumerate(self.spacing):
            el.setAlignment(Qt.AlignRight)
            el.setButtonSymbols(QAbstractSpinBox.NoButtons)
            el.setRange(0, 1000)
            el.setValue(self._settings.image_spacing[i])
        self.units = QComboBox()
        self.units.addItems(UNITS_LIST)
        self.units.setCurrentIndex(2)

        self.add_files = AddFiles(settings, btn_layout=FlowLayout)

        spacing_layout = QFormLayout()
        spacing_layout.addRow("x:", self.spacing[0])
        spacing_layout.addRow("y:", self.spacing[1])
        spacing_layout.addRow("z:", self.spacing[2])
        spacing_layout.addRow("Units:", self.units)

        layout = QVBoxLayout()
        layout.addWidget(self.path)
        layout.addWidget(QLabel("Image spacing:"))
        layout.addLayout(spacing_layout)
        layout.addWidget(self.add_files)
        layout.addStretch()
        self.setLayout(layout)
        self._settings.image_changed[str].connect(self.set_image_path)

    def set_image_path(self, value):
        self.path.setText("<b>Path:</b> {}".format(value))


class Options(QTabWidget):
    def __init__(self, settings, control_view, parent=None):
        super(Options, self).__init__(parent)
        self._settings = settings
        self.algorithm_options = AlgorithmOptions(settings, control_view)
        self.image_properties = ImageInformation(settings, parent)
        self.addTab(self.image_properties, "Image")
        self.addTab(self.algorithm_options, "Segmentation")


class MainWindow(QMainWindow):
    def __init__(self, title):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.settings = ImageSettings()
        self.main_menu = MainMenu(self.settings)
        self.image_view = ImageView(self.settings)
        image_view_control = self.image_view.get_control_view()
        self.options_panel = Options(self.settings, image_view_control)
        self.main_menu.image_loaded.connect(self.image_read)
        self.settings.image_changed.connect(self.image_read)
        self.options_panel.algorithm_options.labels_changed.connect(self.image_view.set_labels)

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
        sub_layout.addWidget(self.options_panel, 0)
        layout.addLayout(sub_layout)
        # self.pixmap.adjustSize()
        # self.pixmap.update_size(2)
        self.widget = QWidget()
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)
        self.settings.image = im

    def image_read(self):
        print("buka1", self.settings.image.shape, self.sender())
        self.image_view.set_image(self.settings.image)




