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
    QLineEdit, QFont, QFrame, QFontMetrics, QMessageBox, QSlider, QCheckBox, QComboBox, QPixmap
from math import copysign


__author__ = "Grzegorz Bokota"



big_font_size = 15
button_margin = 10
button_height = 30
if platform.system() == "Linux":
    big_font_size = 14

if platform.system() == "Darwin":
    big_font_size = 20
    button_margin = 30
    button_height = 40

if platform.system() == "Windows":
    big_font_size = 12


def set_position(elem, previous, dist=10):
    pos_y = previous.y()
    if platform.system() == "Darwin" and isinstance(elem, QLineEdit):
        pos_y += 3
    if platform.system() == "Darwin" and isinstance(previous, QLineEdit):
        pos_y -= 3
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
        text_list = [button.itemText(i) for i in range(2)]
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


class ColormapCanvas(FigureCanvas):
    def __init__(self, figsize, parent):
        fig = pyplot.figure(figsize=figsize, dpi=100, frameon=True, facecolor='1.0', edgecolor='w')
        super(ColormapCanvas, self).__init__(fig)
        self.my_figure_num = fig.number
        self.setParent(parent)
        self.val_min = 0
        self.val_max = 0
        self.color_map = "cubehelix"

    def set_range(self, begin, end=None):
        if end is None and isinstance(begin, np.ndarray):
            self.val_max = begin.max()
            self.val_min = begin.min()
        else:
            self.val_min = begin
            self.val_max = end
        color_map = matplotlib.cm.get_cmap(self.color_map)
        norm = matplotlib.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        fig = pyplot.figure(self.my_figure_num)
        pyplot.clf()
        ax = fig.add_axes([0.05, 0.05, 0.3, 0.9])
        matplotlib.colorbar.ColorbarBase(ax, cmap=color_map, norm=norm, orientation='vertical')
        fig.canvas.draw()


class MyCanvas(FigureCanvas):
    def __init__(self, figsize, parent):
        """
        Create basic canvas to view image
        :param num: Num of figure to use
        :param figsize: Size of figure in inches
        """
        fig = pyplot.figure(figsize=figsize, dpi=100, frameon=True, facecolor='1.0', edgecolor='w')
        super(MyCanvas, self).__init__(fig)
        self.setParent(parent)
        self.my_figure_num = fig.number
        self.toolbar = NavigationToolbar(self, self)
        self.toolbar.hide()
        self.reset_button = QPushButton("Reset zoom", self)
        self.reset_button.move(0, 0)
        self.reset_button.clicked.connect(self.toolbar.home)
        set_button(self.reset_button, None)
        self.zoom_button = QPushButton("Zoom", self)
        self.zoom_button.clicked.connect(self.zoom)
        self.zoom_button.setCheckable(True)
        set_button(self.zoom_button, self.reset_button, -12)
        self.move_button = QPushButton("Move", self)
        set_button(self.move_button, self.zoom_button, -12)
        self.move_button.clicked.connect(self.move_action)
        self.move_button.setCheckable(True)
        self.back_button = QPushButton("Undo", self)
        set_button(self.back_button, self.move_button, -12)
        self.back_button.clicked.connect(self.toolbar.back)
        self.next_button = QPushButton("Redo", self)
        set_button(self.next_button, self.back_button, -12)
        self.next_button.clicked.connect(self.toolbar.forward)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 0)
        self.slider.move(20, self.size().toTuple()[1]-30)
        self.slider.setMinimumWidth(self.width()-80)
        self.layer_num_label = QLabel(self)
        self.layer_num_label.setText("0")
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


class MyDrawCanvas(MyCanvas):
    def __init__(self, figsize, parent):
        super(MyDrawCanvas, self).__init__(figsize, parent)
        self.zoom_button.clicked.connect(self.up_drawing_button)
        self.move_button.clicked.connect(self.up_drawing_button)
        self.draw_button = QPushButton("Draw", self)
        self.draw_button.setCheckable(True)
        set_button(self.draw_button, self.next_button, -12)
        self.draw_button.clicked.connect(self.up_move_zoom_button)
        self.erase_button = QPushButton("Erase", self)
        self.draw_button.setCheckable(True)
        set_button(self.erase_button, self.draw_button, -12)
        self.erase_button.clicked.connect(self.up_move_zoom_button)
        self.clean_button = QPushButton("Clean", self)
        set_button(self.clean_button, self.erase_button, -12)

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


class MainWindow(QMainWindow):
    def __init__(self, title):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.normal_image_canvas = MyCanvas((6, 6), self)
        self.colormap_image_canvas = ColormapCanvas((1, 6), self)
        self.segmented_image_canvas = MyDrawCanvas((6, 6), self)

        self.update_objects_positions()
        self.setGeometry(50, 50,  1400, 720)

    def update_objects_positions(self):
        self.normal_image_canvas.move(10, 40)
        # noinspection PyTypeChecker
        set_position(self.colormap_image_canvas, self.normal_image_canvas, 0)
        # noinspection PyTypeChecker
        set_position(self.segmented_image_canvas, self.colormap_image_canvas, 0)
