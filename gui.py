from __future__ import print_function, division
import sys
import os.path
import os
import tifffile
import SimpleITK as sitk
import numpy as np
import platform
import tempfile
import json
import matplotlib
os.environ['QT_API'] = 'pyside'
matplotlib.use('Qt4Agg')
from matplotlib import pyplot
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from PySide.QtCore import Qt, QSize
from PySide.QtGui import QLabel, QPushButton, QFileDialog, QMainWindow, QStatusBar, QWidget,\
    QLineEdit, QFont, QFrame, QFontMetrics, QMessageBox, QSlider, QCheckBox, QComboBox, QPixmap, QSpinBox,\
    QAbstractSpinBox, QApplication
from math import copysign

from backend import Settings, Segment


__author__ = "Grzegorz Bokota"



big_font_size = 15
button_margin = 10
button_height = 30
button_small_dist = -2

if platform.system() == "Linux":
    big_font_size = 14

if platform.system() == "Darwin":
    big_font_size = 20
    button_margin = 30
    button_height = 40
    button_small_dist = -12

if platform.system() == "Windows":
    big_font_size = 12


def set_position(elem, previous, dist=10):
    pos_y = previous.y()
    if platform.system() == "Darwin" and isinstance(elem, QLineEdit):
        pos_y += 3
    if platform.system() == "Darwin" and isinstance(previous, QLineEdit):
        pos_y -= 3
    if platform.system() == "Darwin" and isinstance(previous, QSlider):
        pos_y -= 10
    if platform.system() == "Darwin" and isinstance(elem, QSpinBox):
            pos_y += 7
    if platform.system() == "Darwin" and isinstance(previous, QSpinBox):
        pos_y -= 7
    elem.move(previous.pos().x() + previous.size().width() + dist, pos_y)


def set_button(button, previous_element, dist=10, super_space=0):
    """
    :type button: QPushButton | QLabel
    :type previous_element: QWidget | None
    :param button:
    :param previous_element:
    :param dist:
    :return:
    """
    font_met = QFontMetrics(button.font())
    if isinstance(button, QComboBox):
        text_list = [button.itemText(i) for i in range(button.count())]
    else:
        text = button.text()
        if text[0] == '&':
            text = text[1:]
        text_list = text.split("\n")
    width = 0
    for txt in text_list:
        width = max(width, font_met.boundingRect(txt).width())
    if isinstance(button, QPushButton):
        button.setFixedWidth(width + button_margin+super_space)
    if isinstance(button, QLabel):
        button.setFixedWidth(width + super_space)
    if isinstance(button, QComboBox):
        button.setFixedWidth(width + button_margin+10)
    button.setFixedHeight(button_height)
    if previous_element is not None:
        set_position(button, previous_element, dist)


def label_to_rgb(image):
    sitk_im = sitk.GetImageFromArray(image)
    lab_im = sitk.LabelToRGB(sitk_im)
    return sitk.GetArrayFromImage(lab_im)


class ColormapCanvas(FigureCanvas):
    def __init__(self, figsize, settings, parent):
        """:type settings: Settings"""
        fig = pyplot.figure(figsize=figsize, dpi=100, frameon=True, facecolor='1.0', edgecolor='w')
        super(ColormapCanvas, self).__init__(fig)
        self.my_figure_num = fig.number
        self.setParent(parent)
        self.val_min = 0
        self.val_max = 0
        self.settings = settings
        settings.add_image_callback(self.set_range)
        settings.add_colormap_callback(self.update_colormap)

    def set_range(self, begin, end=None):
        if end is None and isinstance(begin, np.ndarray):
            self.val_max = begin.max()
            self.val_min = begin.min()
        else:
            self.val_min = begin
            self.val_max = end
        self.update_colormap()

    def update_colormap(self):
        norm = matplotlib.colors.Normalize(vmin=self.val_min, vmax=self.val_max)
        fig = pyplot.figure(self.my_figure_num)
        pyplot.clf()
        ax = fig.add_axes([0.05, 0.08, 0.3, 0.85])
        matplotlib.colorbar.ColorbarBase(ax, cmap=self.settings.color_map, norm=norm, orientation='vertical')
        fig.canvas.draw()


class MyCanvas(FigureCanvas):
    def __init__(self, figsize, settings, parent):
        """
        Create basic canvas to view image
        :param num: Num of figure to use
        :param figsize: Size of figure in inches
        """
        fig = pyplot.figure(figsize=figsize, dpi=100, frameon=True, facecolor='1.0', edgecolor='w')
        super(MyCanvas, self).__init__(fig)
        self.base_image = None
        self.ax_im = None
        self.rgb_image = None
        self.layer_num = 0
        self.setParent(parent)
        self.my_figure_num = fig.number
        self.toolbar = NavigationToolbar(self, self)
        self.toolbar.hide()
        self.reset_button = QPushButton("Reset zoom", self)
        self.reset_button.clicked.connect(self.toolbar.home)
        self.zoom_button = QPushButton("Zoom", self)
        self.zoom_button.clicked.connect(self.zoom)
        self.zoom_button.setCheckable(True)
        self.move_button = QPushButton("Move", self)
        self.move_button.clicked.connect(self.move_action)
        self.move_button.setCheckable(True)
        self.back_button = QPushButton("Undo", self)
        self.back_button.clicked.connect(self.toolbar.back)
        self.next_button = QPushButton("Redo", self)
        self.next_button.clicked.connect(self.toolbar.forward)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 0)
        self.colormap_checkbox = QCheckBox(self)
        self.colormap_checkbox.setText("With colormap")
        self.colormap_checkbox.setChecked(True)
        self.layer_num_label = QLabel(self)
        self.layer_num_label.setText("1 of 1")
        self.settings = settings
        settings.add_image_callback(self.set_image)
        settings.add_colormap_callback(self.update_colormap)
        self.colormap_checkbox.stateChanged.connect(self.update_colormap)
        MyCanvas.update_elements_positions(self)

    def update_elements_positions(self):
        self.reset_button.move(0, 0)
        set_button(self.reset_button, None)
        set_button(self.zoom_button, self.reset_button, button_small_dist)
        set_button(self.move_button, self.zoom_button, button_small_dist)
        set_button(self.back_button, self.move_button, button_small_dist)
        set_button(self.next_button, self.back_button, button_small_dist)
        self.slider.move(20, self.size().toTuple()[1]-20)
        self.colormap_checkbox.move(self.slider.pos().x(), self.slider.pos().y()-15)
        self.slider.setMinimumWidth(self.width()-90)
        set_button(self.layer_num_label, self.slider)

    def move_action(self):
        """
        Deactivate zoom button and call function witch allow moving image
        :return: None
        """
        if self.zoom_button.isChecked():
            self.zoom_button.setChecked(False)
        self.toolbar.pan()

    def zoom(self):
        """
        Deactivate move button and call function witch allow moving image
        :return: None
        """
        if self.move_button.isChecked():
            self.move_button.setChecked(False)
        self.toolbar.zoom()

    def set_image(self, image):
        """
        :type image: np.ndarray
        :return:
        """
        self.base_image = image
        self.ax_im = None
        self.update_rgb_image()
        self.update_image_view()

    def update_colormap(self):
        if self.base_image is None:
            return
        self.update_rgb_image()
        self.update_image_view()

    def update_rgb_image(self):
        float_image = self.base_image / float(self.base_image.max())
        if self.colormap_checkbox.isChecked():
            cmap = self.settings.color_map
        else:
            cmap = matplotlib.cm.get_cmap("gray")
        colored_image = cmap(float_image)
        self.rgb_image = np.array(colored_image * 255).astype(np.uint8)

    def update_image_view(self):
        pyplot.figure(self.my_figure_num)
        if self.base_image.size < 10:
            return
        if len(self.base_image.shape) <= 2:
            image_to_show = self.rgb_image
        else:
            image_to_show = self.rgb_image[self.layer_num]
        if self.ax_im is None:
            pyplot.clf()
            self.ax_im = pyplot.imshow(image_to_show, interpolation='nearest')
        else:
            self.ax_im.set_data(image_to_show)
        self.draw()


class MyDrawCanvas(MyCanvas):
    """
    :type segmentation: np.ndarray
    """
    def __init__(self, figsize, settings,  *args):
        super(MyDrawCanvas, self).__init__(figsize, settings, *args)
        self.draw_canvas = None
        self.history_list = list()
        self.redo_list = list()
        self.zoom_button.clicked.connect(self.up_drawing_button)
        self.move_button.clicked.connect(self.up_drawing_button)
        self.draw_button = QPushButton("Draw", self)
        self.draw_button.setCheckable(True)
        self.draw_button.clicked.connect(self.up_move_zoom_button)
        self.erase_button = QPushButton("Erase", self)
        self.draw_button.setCheckable(True)
        self.erase_button.clicked.connect(self.up_move_zoom_button)
        self.clean_button = QPushButton("Clean", self)
        self.update_elements_positions()
        self.segment = Segment(settings)
        self.segmentation = None
        self.rgb_segmentation = None
        self.original_rgb_image = None
        self.labeled_rgb_image = None
        self.colormap_checkbox.setChecked(False)
        self.segment.add_segmentation_callback(self.segmentation_changed)
        im = [np.arange(3)]
        rgb_im = sitk.GetArrayFromImage(sitk.LabelToRGB(sitk.GetImageFromArray(im)))
        self.draw_colors = rgb_im[0]

    def up_move_zoom_button(self):
        if self.zoom_button.isChecked():
            self.zoom_button.click()
        if self.move_button.isChecked():
            self.move_button.click()

    def up_drawing_button(self):
        # TODO Update after create draw object
        if self.draw_button.isChecked():
            self.draw_button.click()
        if self.erase_button.isChecked():
            self.erase_button.click()

    def update_elements_positions(self):
        super(MyDrawCanvas, self).update_elements_positions()
        set_button(self.draw_button, self.next_button, button_small_dist)
        set_button(self.erase_button, self.draw_button, button_small_dist)
        set_button(self.clean_button, self.erase_button, button_small_dist)

    def segmentation_changed(self):
        self.update_segmentation_rgb()
        self.update_image_view()

    def update_segmentation_rgb(self):
        if self.original_rgb_image is None:
            self.update_rgb_image()
        self.update_segmentation_image()
        mask = self.rgb_segmentation > 0
        overlay = self.settings.overlay
        self.rgb_image = np.copy(self.original_rgb_image)
        self.rgb_image[mask] = self.original_rgb_image[mask] * (1 - overlay) + self.rgb_segmentation[mask] * overlay
        self.labeled_rgb_image = np.copy(self.rgb_image)
        draw_lab = label_to_rgb(self.draw_canvas)
        mask = draw_lab > 0
        self.rgb_image[mask] = self.original_rgb_image[mask] * (1 - overlay) + draw_lab[mask] * overlay

    def update_rgb_image(self):
        super(MyDrawCanvas, self).update_rgb_image()
        self.rgb_image = self.rgb_image[..., :3]
        self.original_rgb_image = np.copy(self.rgb_image)
        self.update_segmentation_rgb()

    def set_image(self, image):
        self.base_image = image
        self.ax_im = None
        self.draw_canvas = np.zeros(image.shape, dtype=np.uint8)
        self.segment.set_image(image)
        self.update_rgb_image()
        self.update_image_view()

    def update_segmentation_image(self):
        if not self.segment.segmentation_changed:
            return
        self.segmentation = np.copy(self.segment.get_segmentation())
        self.segmentation[self.segmentation > 0] += 2
        self.rgb_segmentation = label_to_rgb(self.segmentation)


class MainMenu(QLabel):
    def __init__(self, settings, *args, **kwargs):
        super(MainMenu, self).__init__(*args, **kwargs)
        self.settings = settings
        self.settings.add_image_callback(self.set_threshold_range)
        self.load_button = QPushButton("Load", self)
        self.save_button = QPushButton("Save", self)
        self.save_button.setDisabled(True)
        self.threshold_type = QComboBox(self)
        self.threshold_type.addItem("Upper threshold:")
        self.threshold_type.addItem("Lower threshold:")
        self.threshold_value = QSpinBox(self)
        self.threshold_value.setMinimumWidth(80)
        self.threshold_value.setRange(0, 100000)
        self.threshold_value.setValue(self.settings.threshold)
        self.threshold_value.setSingleStep(500)
        self.threshold_value.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.threshold_value.valueChanged[int].connect(settings.change_threshold)
        self.layer_thr_check = QCheckBox("Layer\nthreshold", self)
        self.minimum_size_lab = QLabel(self)
        self.minimum_size_lab.setText("Minimum\nobject size:")
        self.minimum_size_value = QSpinBox(self)
        self.minimum_size_value.setMinimumWidth(80)
        self.minimum_size_value.setRange(0, 10 ** 6)
        self.minimum_size_value.setValue(self.settings.minimum_size)
        self.minimum_size_value.valueChanged[int].connect(settings.change_min_size)
        self.minimum_size_value.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.minimum_size_value.setSingleStep(10)
        self.gauss_check = QCheckBox("Use gauss", self)
        self.gauss_check.stateChanged[int].connect(settings.change_gauss)
        self.draw_check = QCheckBox("Use draw result", self)

        self.colormap_choose = QComboBox(self)
        self.colormap_choose.addItems(settings.colormap_list)
        index = list(settings.colormap_list).index(settings.color_map_name)
        self.colormap_choose.setCurrentIndex(index)
        self.colormap_choose.currentIndexChanged.connect(self.colormap_changed)

        self.update_elements_positions()
        self.setMinimumWidth(1000)
        self.setMinimumHeight(button_height+5)

    def update_elements_positions(self):
        set_button(self.load_button, None)
        self.load_button.move(0, 0)
        set_button(self.save_button, self.load_button, button_small_dist)
        set_button(self.threshold_type, self.save_button)
        set_position(self.threshold_value, self.threshold_type, 0)
        set_button(self.layer_thr_check, self.threshold_value, -20)
        set_button(self.minimum_size_lab, self.layer_thr_check, 0)
        set_position(self.minimum_size_value, self.minimum_size_lab, 5)
        set_button(self.gauss_check, self.minimum_size_value, -10)
        set_button(self.draw_check, self.gauss_check, -10)
        set_button(self.colormap_choose, self.draw_check, 15)

    def colormap_changed(self):
        self.settings.change_colormap(self.colormap_choose.currentText())

    def settings_changed(self):
        self.threshold_value.setValue(self.settings.threshold)

    def set_threshold_range(self, image):
        vmin = image.min()
        vmax = image.max()
        self.threshold_value.setRange(vmin, vmax)


class MainWindow(QMainWindow):
    def __init__(self, title):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.settings = Settings("settings.json")
        self.main_menu = MainMenu(self.settings, self)

        self.normal_image_canvas = MyCanvas((6, 6), self.settings, self)
        self.colormap_image_canvas = ColormapCanvas((1, 6),  self.settings, self)
        self.segmented_image_canvas = MyDrawCanvas((6, 6), self.settings, self)
        self.segmented_image_canvas.segment.add_segmentation_callback((self.update_object_information,))
        self.slider_swap = QCheckBox("Synchronize\nsliders", self)

        big_font = QFont(QApplication.font())
        big_font.setPointSize(big_font_size)

        self.object_count = QLabel(self)
        self.object_count.setFont(big_font)
        self.object_count.setMinimumWidth(150)
        self.object_size_list = QLabel(self)
        self.object_size_list.setFont(big_font)
        self.object_size_list.setMinimumWidth(1000)
        self.object_size_list.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.setGeometry(50, 50,  1400, 720)

        self.update_objects_positions()
        self.settings.add_image(tifffile.imread("clean_segment.tiff"))

    def update_objects_positions(self):
        self.normal_image_canvas.move(10, 40)
        # noinspection PyTypeChecker
        set_position(self.colormap_image_canvas, self.normal_image_canvas, 0)
        # noinspection PyTypeChecker
        set_position(self.segmented_image_canvas, self.colormap_image_canvas, 0)
        col_pos = self.colormap_image_canvas.pos()
        self.slider_swap.move(col_pos.x()+5,
                              col_pos.y()+self.colormap_image_canvas.height()-35)
        norm_pos = self.normal_image_canvas.pos()
        self.object_count.move(norm_pos.x(),
                               norm_pos.y()+self.normal_image_canvas.height()+20)
        self.object_size_list.move(self.object_count.pos().x()+150, self.object_count.pos().y())

    def update_object_information(self, info_aray):
        """:type info_aray: np.ndarray"""
        self.object_count.setText("Object num: {0}".format(str(info_aray.size)))
        self.object_size_list.setText("Objects size: {0}".format(str(info_aray)))

