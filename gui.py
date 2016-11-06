# coding=utf-8
from __future__ import print_function, division
import os.path
import os
import tifffile
import SimpleITK as sitk
import numpy as np
import platform
import tarfile
import json
import matplotlib
import re
os.environ['QT_API'] = 'pyside'
matplotlib.use('Qt4Agg')
from matplotlib import pyplot
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from PySide.QtCore import Qt, QSize
from PySide.QtGui import QLabel, QPushButton, QFileDialog, QMainWindow, QStatusBar, QWidget,\
    QLineEdit, QFont, QFrame, QFontMetrics, QMessageBox, QSlider, QCheckBox, QComboBox, QPixmap, QSpinBox, \
    QDoubleSpinBox, QAbstractSpinBox, QApplication, QTabWidget, QScrollArea, QInputDialog, QHBoxLayout, QVBoxLayout,\
    QListWidget, QTextEdit

from backend import Settings, Segment, save_to_cmap, save_to_project, load_project


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
    button_height = 34
    button_small_dist = -10

if platform.system() == "Windows":
    big_font_size = 12


def h_line():
    toto = QFrame()
    toto.setFrameShape(QFrame.HLine)
    toto.setFrameShadow(QFrame.Sunken)
    return toto


def v_line():
    toto = QFrame()
    toto.setFrameShape(QFrame.VLine)
    toto.setFrameShadow(QFrame.Sunken)
    return toto


def set_position(elem, previous, dist=10):
    pos_y = previous.pos().y()
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
    :param super_space:
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
    if isinstance(button, QSpinBox):
        button.setAlignment(Qt.AlignRight)
        text_list = [str(button.maximum())+'aa']
        print(text_list)
    width = 0
    for txt in text_list:
        width = max(width, font_met.boundingRect(txt).width())
    if isinstance(button, QPushButton):
        button.setFixedWidth(width + button_margin+super_space)
    if isinstance(button, QLabel):
        button.setFixedWidth(width + super_space)
    if isinstance(button, QComboBox):
        button.setFixedWidth(width + button_margin+10)
    if isinstance(button, QSpinBox):
        print(width)
        button.setFixedWidth(width)
    #button.setFixedHeight(button_height)
    if previous_element is not None:
        set_position(button, previous_element, dist)


def label_to_rgb(image):
    sitk_im = sitk.GetImageFromArray(image)
    lab_im = sitk.LabelToRGB(sitk_im)
    return sitk.GetArrayFromImage(lab_im)


def pack_layout(*args):
    layout = QHBoxLayout()
    layout.setSpacing(0)
    for el in args:
        layout.addWidget(el)
    return layout

class SynchronizeSliders(object):
    def __init__(self, slider1, slider2, switch):
        """
        :type slider1: QSlider
        :type slider2: QSlider
        :type switch: QCheckBox
        """
        self.slider1 = slider1
        self.slider2 = slider2
        self.switch = switch
        self.slider1.valueChanged[int].connect(self.slider1_changed)
        self.slider2.valueChanged[int].connect(self.slider2_changed)
        self.switch.stateChanged[int].connect(self.state_changed)
        self.sync = self.switch.isChecked()

    def state_changed(self, state):
        if state:
            self.slider2.setValue(self.slider1.value())
        self.sync = bool(state)

    def slider1_changed(self, val):
        if not self.sync:
            return
        self.sync = False
        self.slider2.setValue(val)
        self.sync = True

    def slider2_changed(self, val):
        if not self.sync:
            return
        self.sync = False
        self.slider1.setValue(val)
        self.sync = True


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
        self.widget = None
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
        ax = fig.add_axes([0.05, 0.1, 0.3, 0.8])
        matplotlib.colorbar.ColorbarBase(ax, cmap=self.settings.color_map, norm=norm, orientation='vertical')
        fig.canvas.draw()

    def set_widget(self, widget):
        self.widget = widget
        widget.setParent(self)

    def resizeEvent(self, *args, **kwargs):
        super(ColormapCanvas, self).resizeEvent(*args, **kwargs)
        self.widget.move(5, self.height() - 35)


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
        self.slider.valueChanged[int].connect(self.change_layer)
        self.colormap_checkbox = QCheckBox(self)
        self.colormap_checkbox.setText("With colormap")
        self.colormap_checkbox.setChecked(True)
        self.layer_num_label = QLabel(self)
        self.layer_num_label.setText("1 of 1      ")
        self.settings = settings
        settings.add_image_callback(self.set_image)
        settings.add_colormap_callback(self.update_colormap)
        self.colormap_checkbox.stateChanged.connect(self.update_colormap)
        MyCanvas.update_elements_positions(self)
        self.setMinimumHeight(300)

    def update_elements_positions(self):
        self.reset_button.move(0, 0)
        set_button(self.reset_button, None)
        set_button(self.zoom_button, self.reset_button, button_small_dist)
        set_button(self.move_button, self.zoom_button, button_small_dist)
        set_button(self.back_button, self.move_button, button_small_dist)
        set_button(self.next_button, self.back_button, button_small_dist)
        self.slider.move(20, self.size().toTuple()[1]-20)
        self.colormap_checkbox.move(self.slider.pos().x(), self.slider.pos().y()-15)
        self.slider.setMinimumWidth(self.width()-85)
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
        if len(image.shape) > 2:
            self.slider.setRange(0, image.shape[0]-1)
            self.slider.setValue(int(image.shape[0]/2))
        else:
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

    def change_layer(self, layer_num):
        self.layer_num = layer_num
        self.layer_num_label.setText("{0} of {1}".format(str(layer_num + 1), str(self.base_image.shape[0])))
        self.update_image_view()

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

    def resizeEvent(self, resize_event):
        super(MyCanvas, self).resizeEvent(resize_event)
        self.update_elements_positions()


class MyDrawCanvas(MyCanvas):
    """
    :type segmentation: np.ndarray
    """
    def __init__(self, figsize, settings, segment, *args):
        super(MyDrawCanvas, self).__init__(figsize, settings, *args)
        self.draw_canvas = DrawObject(settings, segment, self.update_image_view)
        self.history_list = list()
        self.redo_list = list()
        self.zoom_button.clicked.connect(self.up_drawing_button)
        self.move_button.clicked.connect(self.up_drawing_button)
        self.draw_button = QPushButton("Draw", self)
        self.draw_button.setCheckable(True)
        self.draw_button.clicked.connect(self.up_move_zoom_button)
        self.draw_button.clicked[bool].connect(self.draw_click)
        self.erase_button = QPushButton("Erase", self)
        self.erase_button.setCheckable(True)
        self.erase_button.clicked.connect(self.up_move_zoom_button)
        self.erase_button.clicked[bool].connect(self.erase_click)
        self.clean_button = QPushButton("Clean", self)
        self.clean_button.clicked.connect(self.draw_canvas.clean)
        self.update_elements_positions()
        self.segment = segment
        self.protect_button = False
        self.segmentation = None
        self.rgb_segmentation = None
        self.original_rgb_image = None
        self.labeled_rgb_image = None
        self.colormap_checkbox.setChecked(False)
        self.segment.add_segmentation_callback(self.segmentation_changed)
        self.slider.valueChanged[int].connect(self.draw_canvas.set_layer_num)
        self.slider.valueChanged[int].connect(self.settings.change_layer)
        self.mpl_connect('button_press_event', self.draw_canvas.on_mouse_down)
        self.mpl_connect('motion_notify_event', self.draw_canvas.on_mouse_move)
        self.mpl_connect('button_release_event', self.draw_canvas.on_mouse_up)

    def up_move_zoom_button(self):
        self.protect_button = True
        if self.zoom_button.isChecked():
            self.zoom_button.click()
        if self.move_button.isChecked():
            self.move_button.click()
        self.protect_button = False

    def up_drawing_button(self):
        # TODO Update after create draw object
        if self.protect_button:
            return
        self.draw_canvas.draw_on = False
        if self.draw_button.isChecked():
            self.draw_button.setChecked(False)
        if self.erase_button.isChecked():
            self.erase_button.setChecked(False)

    def draw_click(self, checked):
        if checked:
            self.erase_button.setChecked(False)
            self.draw_canvas.set_draw_mode()
        else:
            self.draw_canvas.draw_on = False

    def erase_click(self, checked):
        if checked:
            self.draw_button.setChecked(False)
            self.draw_canvas.set_erase_mode()
        else:
            self.draw_canvas.draw_on = False

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
        mask = self.segmentation > 0
        overlay = self.settings.overlay
        self.rgb_image = np.copy(self.original_rgb_image)
        self.rgb_image[mask] = self.original_rgb_image[mask] * (1 - overlay) + self.rgb_segmentation[mask] * overlay
        self.labeled_rgb_image = np.copy(self.rgb_image)
        draw_lab = label_to_rgb(self.draw_canvas.draw_canvas)
        if self.settings.use_draw_result:
            mask = (self.draw_canvas.draw_canvas == 1)
            mask *= (self.segmentation == 0)
        else:
            mask = self.draw_canvas.draw_canvas > 0
        self.rgb_image[mask] = self.original_rgb_image[mask] * (1 - overlay) + draw_lab[mask] * overlay
        self.draw_canvas.rgb_image = self.rgb_image
        self.draw_canvas.original_rgb_image = self.original_rgb_image
        self.draw_canvas.labeled_rgb_image = self.labeled_rgb_image

    def update_rgb_image(self):
        super(MyDrawCanvas, self).update_rgb_image()
        self.rgb_image = self.rgb_image[..., :3]
        self.original_rgb_image = np.copy(self.rgb_image)
        self.update_segmentation_rgb()

    def set_image(self, image):
        self.base_image = image
        self.ax_im = None
        self.original_rgb_image = None
        self.draw_canvas.set_image(image)
        self.segment.set_image(image)
        self.update_rgb_image()
        if len(image.shape) > 2:
            self.slider.setRange(0, image.shape[0]-1)
            self.slider.setValue(int(image.shape[0]/2))
        else:
            self.update_image_view()

    def update_segmentation_image(self):
        if not self.segment.segmentation_changed:
            return
        self.segmentation = np.copy(self.segment.get_segmentation())
        self.segmentation[self.segmentation > 0] += 2
        self.rgb_segmentation = label_to_rgb(self.segmentation)


class DrawObject(object):
    def __init__(self, settings, segment, update_fun):
        """
        :type settings: Settings
        :type segment: Segment
        :param update_fun:
        """
        self.settings = settings
        self.segment = segment
        self.mouse_down = False
        self.draw_on = True
        self.draw_canvas = None
        self.prev_x = None
        self.prev_y = None
        self.original_rgb_image = None
        self.rgb_image = None
        self.labeled_rgb_image = None
        self.layer_num = 0
        im = [np.arange(3)]
        rgb_im = sitk.GetArrayFromImage(sitk.LabelToRGB(sitk.GetImageFromArray(im)))
        self.draw_colors = rgb_im[0]
        self.update_fun = update_fun
        self.draw_value = 0
        self.draw_fun = None
        self.value = 1
        self.click_history = []
        self.history = []
        self.f_x = 0
        self.f_y = 0

    def set_layer_num(self, layer_num):
        self.layer_num = layer_num

    def set_draw_mode(self):
        self.draw_on = True
        self.value = 1

    def set_erase_mode(self):
        self.draw_on = True
        self.value = 2

    def set_image(self, image):
        self.draw_canvas = np.zeros(image.shape, dtype=np.uint8)
        self.segment.draw_canvas = self.draw_canvas

    def draw(self, pos):
        if len(self.original_rgb_image.shape) == 3:
            pos = pos[1:]
        self.click_history.append((pos, self.draw_canvas[pos]))
        self.draw_canvas[pos] = self.value
        self.rgb_image[pos] = self.original_rgb_image[pos] * (1-self.settings.overlay) + \
            self.draw_colors[self.value] * self.settings.overlay

    def erase(self, pos):
        if len(self.original_rgb_image.shape) == 3:
            pos = pos[1:]
        self.click_history.append((pos, self.draw_canvas[pos]))
        self.draw_canvas[pos] = 0
        self.rgb_image[pos] = self.labeled_rgb_image[pos]

    def clean(self):
        self.draw_canvas[...] = 0
        self.rgb_image[...] = self.labeled_rgb_image[...]
        self.update_fun()

    def on_mouse_down(self, event):
        self.mouse_down = True
        self.click_history = []
        if self.draw_on and event.xdata is not None and event.ydata is not None:
            ix, iy = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if len(self.original_rgb_image.shape) > 3:
                val = self.draw_canvas[self.layer_num, iy, ix]
            else:
                val = self.draw_canvas[iy, ix]
            if val in [0, self.value]:
                self.draw_fun = self.draw
            else:
                self.draw_fun = self.erase
            self.draw_fun((self.layer_num, iy, ix))
            self.f_x = event.xdata + 0.5
            self.f_y = event.ydata + 0.5
            self.update_fun()

    def on_mouse_move(self, event):
        if self.mouse_down and self.draw_on and event.xdata is not None and event.ydata is not None:
            f_x, f_y = event.xdata + 0.5, event.ydata + 0.5
            # ix, iy = int(f_x), int(f_y)
            max_dist = max(abs(f_x - self.f_x), abs(f_y - self.f_y))
            rep_num = int(max_dist * 2)+2
            points = set()
            for fx, fy in zip(np.linspace(f_x, self.f_x, num=rep_num),np.linspace(f_y, self.f_y, num=rep_num)):
                points.add((int(fx), int(fy)))
            points.remove((int(self.f_x), int(self.f_y)))
            for fx, fy in points:
                self.draw_fun((self.layer_num, fy, fx))
            self.f_x = f_x
            self.f_y = f_y
            self.update_fun()

    def on_mouse_up(self, event):
        self.history.append(self.click_history)
        self.mouse_down = False


class ColormapSettings(QLabel):
    """
    :type cmap_list: list[QCheckBox]
    """
    def __init__(self, settings, parent=None):
        super(ColormapSettings, self).__init__(parent)
        self.accept = QPushButton("Accept", self)
        self.accept.clicked.connect(self.accept_click)
        set_button(self.accept, None)
        self.mark_all = QPushButton("Mark all", self)
        self.mark_all.clicked.connect(self.mark_all_click)
        set_button(self.mark_all, self.accept, button_small_dist)
        self.unselect_all = QPushButton("Unmark all", self)
        self.unselect_all.clicked.connect(self.un_mark_all_click)
        set_button(self.unselect_all, self.mark_all, button_small_dist)
        self.settings = settings
        scroll_area = QScrollArea(self)
        scroll_area.move(0, button_height)
        self.scroll_area = scroll_area
        self.scroll_widget = QLabel()
        self.scroll_area.setWidget(self.scroll_widget)
        choosen = set(settings.colormap_list)
        all_cmap = settings.available_colormap_list
        self.cmap_list = []
        font_met = QFontMetrics(QApplication.font())
        max_len = 0
        for name in all_cmap:
            max_len = max(max_len, font_met.boundingRect(name).width())
            check = QCheckBox(self.scroll_widget)
            check.setText(name)
            if name in choosen:
                check.setChecked(True)
            if name == self.settings.color_map_name:
                check.setDisabled(True)
            self.cmap_list.append(check)
        self.columns = 0
        self.label_len = max_len
        self.update_positions()
        self.settings.add_colormap_callback(self.change_main_colormap)
        self.setMinimumSize(400, 400)

    def mark_all_click(self):
        for elem in self.cmap_list:
            if elem.isEnabled():
                elem.setChecked(True)

    def un_mark_all_click(self):
        for elem in self.cmap_list:
            if elem.isEnabled():
                elem.setChecked(False)

    def accept_click(self):
        choosen = []
        for elem in self.cmap_list:
            if elem.isChecked():
                choosen.append(elem.text())
        self.settings.set_available_colormap(choosen)

    def update_positions(self):
        space = self.size().width()
        space -= 20  # scrollbar
        columns = int(space / float(self.label_len + 10))
        if columns == 0:
            columns = 1
        if columns == self.columns:
            return
        self.columns = columns
        elem = self.cmap_list[0]
        elem.move(0, 0)
        prev = elem
        for count, elem in enumerate(self.cmap_list[1:]):
            if ((count+1) % columns) == 0:
                elem.move(0, prev.pos().y()+20)
            else:
                elem.move(prev.pos().x()+self.label_len+10, prev.pos().y())
            prev = elem
        height = prev.pos().y() + 20
        self.scroll_widget.resize(columns * (self.label_len + 10), height)

    def change_main_colormap(self):
        for elem in self.cmap_list:
            elem.setDisabled(False)
            if elem.text() == self.settings.color_map_name:
                elem.setChecked(True)
                elem.setDisabled(True)

    def resizeEvent(self, resize_event):
        w, h = resize_event.size().toTuple()
        w -= 4
        h -= button_height + 4
        self.scroll_area.resize(w, h)
        self.update_positions()

    def clean(self):
        self.settings.remove_colormap_callback(self.change_main_colormap)


class AdvancedSettings(QLabel):
    def __init__(self, settings, parent=None):
        super(AdvancedSettings, self).__init__(parent)

        def add_label(text, up_layout, widget):
            lab = QLabel(text)
            layout = QHBoxLayout()
            layout.setSpacing(0)
            layout.addWidget(lab)
            layout.addWidget(widget)
            up_layout.addLayout(layout)
            return widget

        def create_spacing(text, layout, num):
            spacing = QSpinBox()
            spacing.setRange(0, 100)
            print(settings.spacing)
            spacing.setValue(settings.spacing[num])
            spacing.setSingleStep(1)
            spacing.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spacing.setAlignment(Qt.AlignRight)
            return add_label(text, layout, spacing)

        def create_voxel_size(text, layout, num):
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0, 1000)
            spinbox.setValue(settings.voxel_size[num])
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(2)
            spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spinbox.setAlignment(Qt.AlignRight)
            return add_label(text, layout, spinbox)

        self.settings = settings
        vlayout = QVBoxLayout()
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Spacing"))
        spacing_layout.addSpacing(11)
        self.x_spacing = create_spacing("x:", spacing_layout, 0)
        self.y_spacing = create_spacing("y:", spacing_layout, 1)
        self.z_spacing = create_spacing("z:", spacing_layout, 2)
        spacing_layout.addStretch()
        vlayout.addLayout(spacing_layout)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Voxel size"))
        self.x_size = create_voxel_size("x:", size_layout, 0)
        self.y_size = create_voxel_size("y:", size_layout, 1)
        self.z_size = create_voxel_size("z:", size_layout, 2)
        self.units_size = QComboBox()
        self.units_size.addItems(["mm", u"µm", "nm", "pm"])
        self.units_size.setCurrentIndex(2)
        for el in [self.x_size, self.y_size, self.z_size]:
            el.valueChanged.connect(self.update_volume)
        self.units_size.currentIndexChanged.connect(self.update_volume)
        size_layout.addWidget(self.units_size)
        vlayout.addLayout(size_layout)
        self.volume_info = QLabel()
        vlayout.addWidget(self.volume_info)

        accept_button = QPushButton("Accept")
        accept_button.clicked.connect(self.accept)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        butt_lay = QHBoxLayout()
        butt_lay.addStretch()
        butt_lay.addWidget(accept_button)
        butt_lay.addWidget(reset_button)
        vlayout.addLayout(butt_lay)
        vlayout.addWidget(h_line())

        profile_lay = QHBoxLayout()
        self.profile_list = QListWidget()
        profile_lay.addWidget(self.profile_list)
        self.profile_list.addItem("<current profile>")
        self.profile_list.addItems(self.settings.profiles.keys())
        self.profile_list.setMaximumWidth(200)
        self.current_profile = QLabel()
        self.current_profile.setWordWrap(True)
        profile_lay.addWidget(self.current_profile)
        text = settings.threshold_type + " threshold: "
        if settings.threshold_layer_separate:
            text += str(settings.threshold_list)
        else:
            text += str(settings.threshold)
        text += "\n"
        text += "Minimum object size: {}\n".format(settings.minimum_size)
        text += "Use gauss [{}]\n".format("x" if settings.use_gauss else " ")
        self.current_profile.setText(text)

        vlayout.addLayout(profile_lay)
        vlayout.addStretch()
        self.setLayout(vlayout)
        self.update_volume()

    def update_volume(self):
        volume = self.x_size.value() * self.y_size.value() * self.z_size.value()
        text = u"Voxel size: {}{}³".format(volume, self.units_size.currentText())
        self.volume_info.setText(text)

    def reset(self):
        self.x_spacing.setValue(self.settings.spacing[0])
        self.y_spacing.setValue(self.settings.spacing[1])
        self.z_spacing.setValue(self.settings.spacing[2])
        self.x_size.setValue(self.settings.voxel_size[0])
        self.y_size.setValue(self.settings.voxel_size[1])
        self.z_size.setValue(self.settings.voxel_size[2])

    def accept(self):
        self.settings.spacing = self.x_spacing.value(), self.y_spacing.value(), self.z_spacing.value()
        self.settings.voxel_size = self.x_size.value(), self.y_size.value(), self.z_size.value()



class AdvancedWindow(QTabWidget):
    def __init__(self, settings, parent=None):
        super(AdvancedWindow, self).__init__(parent)
        self.settings = settings
        self.advanced_settings = AdvancedSettings(settings)
        self.colormap_settings = ColormapSettings(settings)
        self.statistics = QLabel()
        self.addTab(self.advanced_settings, "Settings")
        self.addTab(self.colormap_settings, "Color maps")
        self.addTab(self.statistics, "Statistics")
        if settings.advanced_menu_geometry is not None:
            self.restoreGeometry(settings.advanced_menu_geometry)

    def resizeEvent(self, resize_event):
        super(AdvancedWindow, self).resizeEvent(resize_event)
        """:type new_size: QSize"""
        w, h = resize_event.size().toTuple()
        wt, ht = self.tabBar().size().toTuple()
        h -= ht
        self.colormap_settings.resize(w, h)
        self.statistics.resize(w, h)
        self.advanced_settings.resize(w, h)

    def closeEvent(self, *args, **kwargs):
        self.colormap_settings.clean()
        self.settings.advanced_menu_geometry = self.saveGeometry()
        super(AdvancedWindow, self).closeEvent(*args, **kwargs)


class MainMenu(QLabel):
    def __init__(self, settings, segment, *args, **kwargs):
        super(MainMenu, self).__init__(*args, **kwargs)
        self.settings = settings
        self.segment = segment
        self.settings.add_image_callback(self.set_threshold_range)
        self.settings.add_image_callback(self.set_layer_threshold)
        self.settings.add_change_layer_callback(self.changed_layer)
        self.load_button = QPushButton("Load", self)
        self.load_button.clicked.connect(self.open_file)
        self.save_button = QPushButton("Save", self)
        self.save_button.setDisabled(True)
        self.save_button.clicked.connect(self.save_results)
        self.mask_button = QPushButton("To Mask", self)
        self.mask_button.setDisabled(True)
        self.mask_button.clicked.connect(self.segmentation_to_mask)
        self.threshold_type = QComboBox(self)
        self.threshold_type.addItem("Upper threshold:")
        self.threshold_type.addItem("Lower threshold:")
        self.threshold_type.currentIndexChanged[unicode].connect(settings.change_threshold_type)
        self.threshold_value = QSpinBox(self)
        self.threshold_value.setMinimumWidth(80)
        self.threshold_value.setRange(0, 100000)
        self.threshold_value.setAlignment(Qt.AlignRight)
        self.threshold_value.setValue(self.settings.threshold)
        self.threshold_value.setSingleStep(500)
        self.threshold_value.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.threshold_value.valueChanged[int].connect(settings.change_threshold)
        self.layer_thr_check = QCheckBox("Layer\nthreshold", self)
        self.layer_thr_check.clicked[bool].connect(self.settings.change_layer_threshold)
        self.minimum_size_lab = QLabel(self)
        self.minimum_size_lab.setText("Minimum object size:")
        self.minimum_size_value = QSpinBox(self)
        self.minimum_size_value.setMinimumWidth(60)
        self.minimum_size_value.setAlignment(Qt.AlignRight)
        self.minimum_size_value.setRange(0, 10 ** 6)
        self.minimum_size_value.setValue(self.settings.minimum_size)
        self.minimum_size_value.valueChanged[int].connect(settings.change_min_size)
        self.minimum_size_value.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.minimum_size_value.setSingleStep(10)
        self.gauss_check = QCheckBox("Use gauss", self)
        self.gauss_check.stateChanged[int].connect(settings.change_gauss)
        self.draw_check = QCheckBox("Use draw\n result", self)
        self.draw_check.stateChanged[int].connect(settings.change_draw_use)
        self.profile_choose = QComboBox(self)
        self.profile_choose.addItem("<no profile>")
        self.profile_choose.addItems(self.settings.get_profile_list())
        self.advanced_button = QPushButton("Advanced", self)
        self.advanced_button.clicked.connect(self.open_advanced)
        self.advanced_window = None

        self.colormap_choose = QComboBox(self)
        self.colormap_choose.addItems(sorted(settings.colormap_list, key=lambda x: x.lower()))
        index = sorted(settings.colormap_list, key=lambda x: x.lower()).index(settings.color_map_name)
        self.colormap_choose.setCurrentIndex(index)
        self.colormap_choose.currentIndexChanged.connect(self.colormap_changed)
        self.settings.add_colormap_list_callback(self.colormap_list_changed)
        self.colormap_protect = False
        #self.setMinimumWidth(1200)
        self.setMinimumHeight(50)
        self.update_elements_positions()
        self.one_line = True

    def update_elements_positions(self):
        layout = QHBoxLayout()
        second_list = [self.gauss_check, self.draw_check, self.profile_choose,
                       self.advanced_button, self.colormap_choose]
        layout.addLayout(pack_layout(self.load_button, self.save_button, self.mask_button))
        # layout.addWidget(self.load_button)
        # layout.addWidget(self.save_button)
        # layout.addWidget(self.export_button)
        layout.addLayout(pack_layout(self.threshold_type, self.threshold_value))
        # layout.addWidget(self.threshold_type)
        # layout.addWidget(self.threshold_value)
        layout.addWidget(self.layer_thr_check)
        layout.addLayout(pack_layout(self.minimum_size_lab, self.minimum_size_value))
        # layout.addWidget(self.minimum_size_lab)
        # layout.addWidget(self.minimum_size_value)
        if False and self.size().width() < 1200:
            print("two")
            layout.addStretch()
            vertical = QVBoxLayout()
            layout2 = QHBoxLayout()
            for el in second_list:
                layout2.addWidget(el)
            layout2.addStretch()
            vertical.addLayout(layout)
            vertical.addLayout(layout2)
            self.setLayout(vertical)
        else:
            print("one")
            for el in second_list:
                layout.addWidget(el)
            layout.addStretch()
            self.setLayout(layout)

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def resizeEvent(self, resize):
        super(MainMenu, self).resizeEvent(resize)
        print(self.size())
        if self.size().width() < 1200 and self.one_line:
            self.one_line = False
            print("buak")
        if self.size().width() >= 1200 and not self.one_line:
            self.one_line = True
            print("buka")

    def colormap_changed(self):
        if self.colormap_protect:
            return
        self.settings.change_colormap(self.colormap_choose.currentText())

    def settings_changed(self):
        self.segment.protect = True
        self.minimum_size_value.setValue(self.settings.minimum_size)
        if self.settings.threshold_layer_separate:
            self.threshold_value.setValue(
                self.settings.threshold_list[self.settings.layer_num])
        else:
            self.threshold_value.setValue(self.settings.threshold)
        self.gauss_check.setChecked(self.settings.use_gauss)
        self.draw_check.setChecked(self.settings.use_draw_result)
        self.segment.protect = False

    def colormap_list_changed(self):
        self.colormap_protect = True
        text = self.colormap_choose.currentText()
        self.colormap_choose.clear()
        self.colormap_choose.addItems(self.settings.colormap_list)
        index = list(self.settings.colormap_list).index(text)
        self.colormap_choose.setCurrentIndex(index)
        self.colormap_protect = False

    def set_threshold_range(self, image):
        val_min = image.min()
        val_max = image.max()
        self.threshold_value.setRange(val_min, val_max)
        diff = val_max - val_min
        if diff > 10000:
            self.threshold_value.setSingleStep(500)
        elif diff > 1000:
            self.threshold_value.setSingleStep(100)
        elif diff > 100:
            self.threshold_value.setSingleStep(20)
        else:
            self.threshold_value.setSingleStep(1)

    def set_layer_threshold(self, *args):
        self.layer_thr_check.setChecked(False)

    def changed_layer(self, lay_num):
        self.threshold_value.setValue(lay_num)

    def segmentation_to_mask(self):
        msgbox = QMessageBox(self)
        msgbox.setText("Change segmentation on mask")
        msgbox.setInformativeText("This change can not be undone")
        msgbox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msgbox.setDefaultButton(QMessageBox.Ok)
        ret = msgbox.exec_()
        if ret == QMessageBox.Ok:
            self.settings.mask = self.segment.get_segmentation()

    def open_file(self):
        dial = QFileDialog(self, "Load data")
        if self.settings.open_directory is not None:
            dial.setDirectory(self.settings.open_directory)
        dial.setFileMode(QFileDialog.ExistingFile)
        filters = ["raw image (*.tiff *.tif *.lsm)", "image with mask (*.tiff *.tif *.lsm *json)",
                   "saved project (*.gz)"]
        dial.setFilters(filters)
        if self.settings.open_filter is not None:
            dial.selectNameFilter(self.settings.open_filter)
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self.settings.open_directory = os.path.dirname(file_path)
            selected_filter = dial.selectedFilter()
            self.settings.open_filter = selected_filter
            print(file_path, selected_filter)
            # TODO maybe something better. Now main window have to be parent
            if selected_filter == "raw image (*.tiff *.tif *.lsm)":
                im = tifffile.imread(file_path)
                if len(im.shape) == 4:
                    # TODO do something better. now not all possibilities are covered
                    num, state = QInputDialog.getInt(self, "Get channel number", "Witch channel:", 0, 0, im.shape[-1])
                    if state:
                        im = im[..., num]
                    else:
                        return
                self.settings.add_image(im, file_path)
            elif selected_filter == "image with mask (*.tiff *.tif *.lsm *json)":
                extension = os.path.splitext(file_path)
                if extension == ".json":
                    with open(file_path) as ff:
                        info_dict = json.load(ff)
                    image = tifffile.imread(info_dict["image"])
                    mask = tifffile.imread(info_dict["mask"])
                    self.settings.add_image(image, file_path, mask)
                else:
                    image = tifffile.imread(file_path)
                    mask_dial = QFileDialog(self, "Load mask")
                    filters = ["mask (*.tiff *.tif *.lsm)"]
                    mask_dial.setFilters(filters)
                    if mask_dial.exec_():
                        mask = tifffile.imread(mask_dial.selectedFiles()[0])
                        self.settings.add_image(image, file_path, mask)
            elif selected_filter == "saved project (*.gz)":
                load_project(file_path,self.settings, self.segment)
                self.settings_changed()
                # self.segment.threshold_updated()
            else:
                r = QMessageBox.warning(self, "Load error", "Function do not implemented yet")
                return
            self.save_button.setEnabled(True)
            self.mask_button.setEnabled(True)

    def save_results(self):
        dial = QFileDialog(self, "Save data")
        if self.settings.save_directory is not None:
            dial.setDirectory(self.settings.save_directory)
        dial.setFileMode(QFileDialog.AnyFile)
        filters = ["Project (*.gz)", "Labeled image (*.tif)", "Mask in tiff (*.tif)",
                   "Mask for itk-snap (*.img)", "Data for chimera (*.cmap)", "Image (*.tiff)"]
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setFilters(filters)
        deafult_name = os.path.splitext(os.path.basename(self.settings.file_path))[0]
        dial.selectFile(deafult_name)
        if self.settings.save_filter is not None:
            dial.selectNameFilter(self.settings.save_filter)
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            selected_filter = dial.selectedFilter()
            self.settings.save_filter = selected_filter
            self.settings.save_directory = os.path.dirname(file_path)
            if os.path.splitext(file_path)[1] == '':
                ext = re.search(r'\(\*(\.\w+)\)', selected_filter).group(1)
                file_path += ext
            if selected_filter == "Project (*.gz)":
                save_to_project(file_path,self.settings, self.segment)

            elif selected_filter == "Labeled image (*.tif)":
                segmentation = self.segment.get_segmentation()
                image = np.copy(self.settings.image)
                cmap = matplotlib.cm.get_cmap("gray")
                float_image = image / float(image.max())
                rgb_image = cmap(float_image)
                label_image = sitk.GetArrayFromImage(sitk.LabelToRGB(sitk.GetImageFromArray(segmentation)))
                rgb_image = (rgb_image[...,:3] * 256).astype(np.uint8)
                mask = segmentation >0
                overlay = self.settings.overlay
                rgb_image[mask] = rgb_image[mask] * (1 - overlay) + label_image[mask] * overlay
                tifffile.imsave(file_path, rgb_image)

            elif selected_filter == "Mask in tiff (*.tif)":
                segmentation = self.segment.get_segmentation()
                tifffile.imsave(file_path, segmentation)
            elif selected_filter == "Mask for itk-snap (*.img)":
                segmentation = sitk.GetImageFromArray(self.segment.get_segmentation())
                sitk.WriteImage(segmentation, file_path)
            elif selected_filter == "Data for chimera (*.cmap)":
                save_to_cmap(file_path,self.settings, self.segment)
            elif selected_filter == "Image (*.tiff)":
                image = self.settings.image
                tifffile.imsave(file_path, image)
            else:
                r = QMessageBox.critical(self, "Save error", "Option unknow")

    def open_advanced(self):
        self.advanced_window = AdvancedWindow(self.settings)
        print(self.settings.spacing)
        self.advanced_window.show()


class MainWindow(QMainWindow):
    def __init__(self, title):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.settings = Settings("settings.json")
        self.segment = Segment(self.settings)
        self.main_menu = MainMenu(self.settings, self.segment, self)

        self.normal_image_canvas = MyCanvas((6, 6), self.settings, self)
        self.colormap_image_canvas = ColormapCanvas((1, 6),  self.settings, self)
        self.segmented_image_canvas = MyDrawCanvas((6, 6), self.settings, self.segment, self)
        self.segmented_image_canvas.segment.add_segmentation_callback((self.update_object_information,))
        self.slider_swap = QCheckBox("Synchronize\nsliders", self)
        self.sync = SynchronizeSliders(self.normal_image_canvas.slider, self.segmented_image_canvas.slider,
                                       self.slider_swap)
        self.colormap_image_canvas.set_widget(self.slider_swap)

        #self.infoText = QLabel("Bright: 0\nComp:", self)

        big_font = QFont(QApplication.font())
        big_font.setPointSize(big_font_size)

        self.object_count = QLabel(self)
        self.object_count.setFont(big_font)
        self.object_count.setFixedWidth(150)
        self.object_size_list = QTextEdit(self)
        self.object_size_list.setReadOnly(True)
        self.object_size_list.setFont(big_font)
        self.object_size_list.setMinimumWidth(800)
        self.object_size_list.setMaximumHeight(30)
        # self.object_size_list.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # self.object_size_list_area.setWidget(self.object_size_list)
        # self.object_size_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.object_size_list.setMinimumHeight(200)
        # self.object_size_list.setWordWrap(True)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.settings.add_image_callback((self.statusBar.showMessage, str))

        self.setGeometry(50, 50,  1400, 720)

        self.update_objects_positions()
        self.settings.add_image(tifffile.imread("clean_segment.tiff"), "")

    def update_objects_positions(self):
        widget = QWidget()
        main_layout = QVBoxLayout()
        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.main_menu)
        # menu_layout.addStretch()
        main_layout.addLayout(menu_layout)
        image_layout = QHBoxLayout()
        image_layout.setSpacing(0)
        image_layout.addWidget(self.normal_image_canvas)
        image_layout.addWidget(self.colormap_image_canvas)
        image_layout.addWidget(self.segmented_image_canvas)
        image_layout.addStretch()
        main_layout.addLayout(image_layout)
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.object_count)
        info_layout.addWidget(self.object_size_list)
        main_layout.addLayout(info_layout)
        main_layout.addStretch()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def resizeEvent(self, *args, **kwargs):
        super(MainWindow, self).resizeEvent(*args, **kwargs)

    def update_objects_positions2(self):
        self.normal_image_canvas.move(10, 40)
        # noinspection PyTypeChecker
        set_position(self.colormap_image_canvas, self.normal_image_canvas, 0)
        # noinspection PyTypeChecker
        set_position(self.segmented_image_canvas, self.colormap_image_canvas, 0)
        col_pos = self.colormap_image_canvas.pos()
        self.slider_swap.move(col_pos.x()+5,
                              col_pos.y()+self.colormap_image_canvas.height()-35)
        # self.infoText.move()

        norm_pos = self.normal_image_canvas.pos()
        self.object_count.move(norm_pos.x(),
                               norm_pos.y()+self.normal_image_canvas.height()+20)
        self.object_size_list.move(self.object_count.pos().x()+150, self.object_count.pos().y())

    def update_object_information(self, info_aray):
        """:type info_aray: np.ndarray"""
        self.object_count.setText("Object num: {0}".format(str(info_aray.size)))
        self.object_size_list.setText("Objects size: {0}".format(str(info_aray)))

    def closeEvent(self, event):
        self.settings.dump("settings.json")
