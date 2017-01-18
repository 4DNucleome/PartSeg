# coding=utf-8
from __future__ import print_function, division
import os.path
import os
import tifffile
import SimpleITK as sitk
import numpy as np
import json
import re
import sys
import appdirs
from qt_import import *
from global_settings import file_folder

from matplotlib import pyplot
import matplotlib.colors as colors

from PIL import Image

from backend import Settings, Segment, UPPER, GAUSS, SegmentationProfile, MaskChange
from batch_window import BatchWindow

from image_operations import DrawType

from io_functions import save_to_cmap, save_to_project, load_project, GaussUse

from advanced_window import AdvancedWindow

__author__ = "Grzegorz Bokota"

app_name = "PartSeg"
app_lab = "LFSG"

canvas_icon_size = QSize(27, 27)

reaction_time = 500

config_folder = appdirs.user_data_dir(app_name, app_lab)
print (config_folder)

if not os.path.isdir(config_folder):
    os.makedirs(config_folder)


def label_to_rgb(image):
    sitk_im = sitk.GetImageFromArray(image)
    lab_im = sitk.LabelToRGB(sitk_im)
    return sitk.GetArrayFromImage(lab_im)


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


class ColormapCanvas(QWidget):
    """
    :type settings: Settings
    """
    def __init__(self, figure_size, settings, parent):
        """:type settings: Settings"""
        super(ColormapCanvas, self).__init__(parent)
        fig = pyplot.figure(figsize=figure_size, dpi=100, frameon=False, facecolor='1.0', edgecolor='w')
        self.figure_canvas = FigureCanvas(fig)
        self.my_figure_num = fig.number
        self.setParent(parent)
        self.val_min = 0
        self.val_max = 0
        self.settings = settings
        self.bottom_widget = None
        self.top_widget = None
        self.slider = QSlider(self)
        self.slider.setRange(-10000, -50)
        # self.slider.setValue(-int(100/e))
        self.slider.setSingleStep(1)
        self.slider.valueChanged[int].connect(self.slider_changed)
        self.slider.setOrientation(Qt.Vertical)
        self.norm = QDoubleSpinBox(self)
        self.norm.setRange(0.01, 10)
        self.norm.setSingleStep(0.1)
        self.norm.valueChanged[float].connect(self.value_changed)
        self.protect = False
        self.starting_value = -5000
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_update)
        self.timer.setSingleShot(True)
        settings.add_image_callback(self.set_range)
        settings.add_colormap_callback(self.update_colormap)
        settings.add_metadata_changed_callback(self.update_colormap)
        self.timer.timeout.connect(self.settings.change_colormap)

    def value_changed(self, val):
        if self.protect:
            return
        self.settings.power_norm = val
        self.timer.stop()
        self.timer.start(reaction_time)

    def slider_changed(self, val):
        if self.protect:
            return
        val = -val

        if val < 5000:
            real_val = val / 5000.0
        else:
            real_val = 1 + (val - 5000)/5000 * 9
        # print(val, real_val)
        self.settings.power_norm = real_val
        self.protect = True
        self.norm.setValue(self.settings.power_norm)
        self.protect = False
        # self.update_colormap()
        if abs(self.starting_value - val) > 500:
            self.run_update(True)
            self.starting_value = val
            self.timer.stop()
        else:
            if not self.timer.isActive():
                self.timer.start(2*reaction_time)

    def run_update(self, manual=False):
        # print("Update manual {}".format(manual))
        self.protect = True
        self.settings.change_colormap()
        self.protect = False

    def set_range(self, begin, end=None):
        if end is None and isinstance(begin, np.ndarray):
            self.val_max = begin.max()
            self.val_min = begin.min()
        else:
            self.val_min = begin
            self.val_max = end
        self.update_colormap()

    def update_colormap(self):
        self.norm.setValue(self.settings.power_norm)
        if self.settings.power_norm <= 1:
            new_val = int(self.settings.power_norm * 5000)
        else:
            new_val = (5000 + int(5000/9 * (self.settings.power_norm - 1)))
        self.slider.setValue(-new_val)
        self.starting_value = new_val
        norm = colors.PowerNorm(gamma=self.settings.power_norm, vmin=self.val_min, vmax=self.val_max)
        fig = pyplot.figure(self.my_figure_num)
        pyplot.clf()
        ax = fig.add_axes([0.01, 0.01, 0.25, 0.98])
        matplotlib.colorbar.ColorbarBase(ax, cmap=self.settings.color_map, norm=norm, orientation='vertical')
        fig.canvas.draw()

    def set_bottom_widget(self, widget):
        self.bottom_widget = widget
        widget.setParent(self)

    def set_top_widget(self, widget):
        self.top_widget = widget
        widget.setParent(self)

    def set_layout(self):
        layout = QVBoxLayout()
        layout.setSpacing(0)
        if self.top_widget is not None:
            h_layout = QHBoxLayout()
            h_layout.addWidget(self.top_widget)
            layout.addLayout(h_layout)
        mid = QHBoxLayout()
        mid.addWidget(self.slider)
        mid.addWidget(self.figure_canvas)
        layout.addWidget(self.norm)
        layout.addLayout(mid)
        #layout.addWidget(self.figure_canvas)
        layout.setStretchFactor(self.figure_canvas, 1)
        if self.bottom_widget is not None:
            layout.addWidget(self.bottom_widget)
        self.setLayout(layout)

    def resizeEvent(self, *args, **kwargs):
        super(ColormapCanvas, self).resizeEvent(*args, **kwargs)
        """if self.bottom_widget is not None:
            self.bottom_widget.move(5, self.height() - 35)
        if self.top_widget is not None:
            self.top_widget.move(5, 5)"""


class CropSet(QDialog):
    def __init__(self, max_size, current_size):
        super(CropSet, self).__init__()
        self.max_size = max_size
        self.min_x = QDoubleSpinBox(self)
        self.max_x = QDoubleSpinBox(self)
        self.min_y = QDoubleSpinBox(self)
        self.max_y = QDoubleSpinBox(self)
        self.min_x.setDecimals(3)
        self.max_x.setDecimals(3)
        self.min_y.setDecimals(3)
        self.max_y.setDecimals(3)
        self.min_x.setRange(0, max_size[0])
        self.min_x.setValue(current_size[0][0])
        self.max_x.setRange(0, max_size[0])
        self.max_x.setValue(current_size[0][1])
        self.min_y.setRange(0, max_size[1])
        self.min_y.setValue(current_size[1][0])
        self.max_y.setRange(0, max_size[1])
        self.max_y.setValue(current_size[1][1])
        self.min_x.valueChanged[float].connect(self.min_x_changed)
        self.max_x.valueChanged[float].connect(self.max_x_changed)
        self.min_y.valueChanged[float].connect(self.min_y_changed)
        self.max_y.valueChanged[float].connect(self.max_y_changed)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Image size: {}".format(max_size)))
        range_layout = QGridLayout()
        range_layout.addWidget(QLabel("Width:"), 0, 0)
        x_layout = QHBoxLayout()
        x_layout.addWidget(self.min_x)
        x_layout.addWidget(self.max_x)
        range_layout.addLayout(x_layout, 0, 1)
        range_layout.addWidget(QLabel("Height"), 1, 0)
        y_layout = QHBoxLayout()
        y_layout.addWidget(self.min_y)
        y_layout.addWidget(self.max_y)
        range_layout.addLayout(y_layout, 1, 1)
        layout.addLayout(range_layout)
        button_layout = QHBoxLayout()
        close_button = QPushButton("Cancel")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        save_button = QPushButton("Crop image")
        save_button.clicked.connect(self.accept)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def min_x_changed(self, val):
        self.max_x.setMinimum(val + 1)

    def max_x_changed(self, val):
        self.min_x.setMaximum(val - 1)

    def min_y_changed(self, val):
        self.max_y.setMinimum(val + 1)

    def max_y_changed(self, val):
        self.min_y.setMaximum(val - 1)

    def get_range(self):
        return (self.min_x.value(), self.max_x.value()), (self.min_y.value(), self.max_y.value())


class MyCanvas(QWidget):
    """
    :type settings: Settings
    """
    def __init__(self, figure_size, settings, info_object, parent, settings_callback=True):
        """
        Create basic canvas to view image
        :param figure_size: Size of figure in inches
        """

        fig = pyplot.figure(figsize=figure_size, dpi=100, frameon=False, facecolor='1.0', edgecolor='w',
                            tight_layout=True)
        # , tight_layout=tight_dict)
        super(MyCanvas, self).__init__(parent)
        self.settings = settings
        self.info_object = info_object

        self.figure_canvas = FigureCanvas(fig)
        self.base_image = None
        self.gauss_image = None
        self.max_value = 1
        self.min_value = 0
        self.ax_im = None
        self.rgb_image = None
        self.layer_num = 0
        self.main_layout = None
        self.zoom_sync = False
        self.sync_fig_num = 0

        # self.setParent(parent)
        self.my_figure_num = fig.number
        self.toolbar = NavigationToolbar(self.figure_canvas, self)
        self.toolbar.hide()
        self.reset_button = QToolButton(self)
        self.reset_button.setIcon(QIcon(os.path.join(file_folder, "icons", "zoom-original.png")))
        self.reset_button.setIconSize(canvas_icon_size)
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setToolTip("Reset zoom")
        self.zoom_button = QToolButton(self)
        self.zoom_button.setIcon(QIcon(os.path.join(file_folder, "icons", "zoom-select.png")))
        self.zoom_button.setIconSize(canvas_icon_size)
        self.zoom_button.setToolTip("Zoom")
        self.zoom_button.clicked.connect(self.zoom)
        self.zoom_button.setCheckable(True)
        self.zoom_button.setContextMenuPolicy(Qt.ActionsContextMenu)
        crop = QAction("Crop", self.zoom_button)
        crop.triggered.connect(self.crop_view)
        self.zoom_button.addAction(crop)
        self.move_button = QToolButton(self)
        self.move_button.setToolTip("Move")
        self.move_button.setIcon(QIcon(os.path.join(file_folder, "icons", "transform-move.png")))
        self.move_button.setIconSize(canvas_icon_size)
        self.move_button.clicked.connect(self.move_action)
        self.move_button.setCheckable(True)
        # self.back_button = QPushButton("Undo", self)
        # noinspection PyUnresolvedReferences
        # self.back_button.clicked.connect(self.toolbar.back)
        # self.next_button = QPushButton("Redo", self)
        # self.next_button.clicked.connect(self.toolbar.forward)
        self.button_list = [self.reset_button, self.zoom_button, self.move_button] #, self.back_button, self.next_button]
        self.mouse_pressed = False
        self.begin_pos = None
        self.last_pos = None

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 0)
        self.slider.valueChanged[int].connect(self.change_layer)
        self.colormap_checkbox = QCheckBox(self)
        self.colormap_checkbox.setText("With colormap")
        self.colormap_checkbox.setChecked(True)
        self.mark_mask = QCheckBox(self)
        self.mark_mask.setText("Mark mask")
        self.mark_mask.setChecked(False)
        self.gauss_view = QCheckBox(self)
        self.gauss_view.setText("Gauss image")
        self.gauss_view.setChecked(False)
        # self.mark_mask.setDisabled(True)
        self.layer_num_label = QLabel(self)
        self.layer_num_label.setText("1 of 1      ")
        self.colormap_checkbox.stateChanged.connect(self.update_colormap)
        self.mark_mask.stateChanged.connect(self.update_colormap)
        self.gauss_view.stateChanged.connect(self.update_colormap)
        if settings_callback:
            settings.add_image_callback((self.set_image, GAUSS))
            settings.add_colormap_callback(self.update_colormap)
            self.settings.add_threshold_type_callback(self.update_colormap)
        # MyCanvas.update_elements_positions(self)
        # self.setFixedWidth(500)
        self.figure_canvas.mpl_connect('button_release_event', self.zoom_sync_fun)
        self.figure_canvas.mpl_connect('button_release_event', self.move_sync_fun)
        self.figure_canvas.mpl_connect('button_release_event', self.mouse_up)
        self.figure_canvas.mpl_connect('button_press_event', self.mouse_down)
        self.figure_canvas.mpl_connect('motion_notify_event', self.brightness_up)
        self.figure_canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.figure_canvas.mpl_connect('scroll_event', self.zoom_scale)

    def crop_view(self):
        shape = self.base_image.shape
        shape = shape[-1], shape[-2]
        fig = pyplot.figure(self.my_figure_num)
        xlim = pyplot.xlim()
        ylim = pyplot.ylim()
        cr = CropSet(shape, ((xlim[0] + 0.5, xlim[1] + 0.5), (ylim[1] + 0.5, ylim[0] + 0.5)))
        if cr.exec_():
            res = cr.get_range()
            logging.debug("crop {}".format(res))
            pyplot.xlim(res[0][0] - 0.5, res[0][1] - 0.5)
            pyplot.ylim(res[1][1] - 0.5, res[1][0] - 0.5)
            fig.canvas.draw()

    def zoom_scale(self, event):
        if self.zoom_button.isChecked() or self.move_button.isChecked():
            scale_factor = self.settings.scale_factor
            if event.button == "down":
                scale_factor = 1 / scale_factor

            def new_pos(mid, pos):
                return mid - (mid - pos) * scale_factor

            fig = pyplot.figure(self.my_figure_num)
            ax_size = pyplot.xlim()
            ay_size = pyplot.ylim()
            ax_size_n = (new_pos(event.xdata, ax_size[0]), new_pos(event.xdata, ax_size[1]))
            ay_size_n = (new_pos(event.ydata, ay_size[0]), new_pos(event.ydata, ay_size[1]))

            pyplot.xlim(ax_size_n)
            pyplot.ylim(ay_size_n)
            fig.canvas.draw()
            if self.zoom_sync:
                fig = pyplot.figure(self.sync_fig_num)
                pyplot.xlim(ax_size_n)
                pyplot.ylim(ay_size_n)
                fig.canvas.draw()

    def brightness_up(self, event):
        if self.info_object is None:
            return
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if self.gauss_view.isChecked():
                img = self.gauss_image
            else:
                img = self.base_image
            """:type img: np.ndarray"""
            try:
                if img.ndim == 2:
                    self.info_object.update_brightness(img[y, x])
                else:
                    self.info_object.update_brightness(img[self.layer_num, y, x])
            except IndexError:
                pass
        else:
            self.info_object.update_brightness(None)

    def mouse_up(self, _):
        self.mouse_pressed = False

    def mouse_down(self, event):
        self.mouse_pressed = True
        if event.xdata is not None and event.ydata is not None:
            self.begin_pos = event.xdata, event.ydata + 0.5
            self.last_pos = self.begin_pos
        else:
            self.begin_pos = None
            self.last_pos = None

    def mouse_move(self, event):
        if self.last_pos is None:
            return
        if event.xdata is not None:
            self.last_pos = event.xdata, self.last_pos[1]
        if event.ydata is not None:
            self.last_pos = self.last_pos[0], event.ydata

    def zoom_sync_fun(self, _):
        if self.zoom_sync and self.zoom_button.isChecked():
            fig = pyplot.figure(self.sync_fig_num)
            x_size = self.begin_pos[0], self.last_pos[0]
            if x_size[0] > x_size[1]:
                x_size = x_size[1], x_size[0]
            y_size = self.begin_pos[1], self.last_pos[1]
            if y_size[0] < y_size[1]:
                y_size = y_size[1], y_size[0]
            pyplot.xlim(x_size)
            pyplot.ylim(y_size)
            fig.canvas.draw()

    def move_sync_fun(self, _):
        if self.zoom_sync and self.move_button.isChecked():
            pyplot.figure(self.my_figure_num)
            ax_size = pyplot.xlim()
            ay_size = pyplot.ylim()
            fig = pyplot.figure(self.sync_fig_num)
            pyplot.xlim(ax_size)
            pyplot.ylim(ay_size)
            fig.canvas.draw()

    def update_elements_positions(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0, 0, 0, 0)
        for butt in self.button_list:
            button_layout.addWidget(butt)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.figure_canvas)
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_layout.addWidget(self.colormap_checkbox)
        checkbox_layout.addWidget(self.mark_mask)
        checkbox_layout.addWidget(self.gauss_view)
        checkbox_layout.setSpacing(10)
        checkbox_layout.addStretch()
        main_layout.addLayout(checkbox_layout)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_layout.addSpacing(15)
        slider_layout.addWidget(self.layer_num_label)
        main_layout.addLayout(slider_layout)
        self.setLayout(main_layout)
        self.main_layout = main_layout
        # print(self.__class__.__name__, "Spacing", main_layout.spacing(), button_layout.spacing())

    def sync_zoom(self, state):
        self.zoom_sync = state

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

    def reset(self):
        # self.toolbar.home()
        fig = pyplot.figure(self.my_figure_num)
        ax_size = (-0.5, self.base_image.shape[-1] - 0.5)
        ay_size = (self.base_image.shape[-2] - 0.5, -0.5)
        pyplot.xlim(ax_size)
        pyplot.ylim(ay_size)
        fig.canvas.draw()
        if self.zoom_sync:
            fig = pyplot.figure(self.sync_fig_num)
            pyplot.xlim(ax_size)
            pyplot.ylim(ay_size)
            fig.canvas.draw()

    def set_image(self, image, gauss, image_update=False):
        """
        :type image: np.ndarray
        :type gauss: np.ndarray
        :type image_update: bool
        :return:
        """
        self.base_image = image
        pyplot.figure(self.my_figure_num)
        ax_lim = pyplot.xlim()
        ay_lim = pyplot.ylim()
        self.max_value = image.max()
        self.min_value = image.min()
        self.gauss_image = gauss
        if gauss is None:
            self.gauss_view.setDisabled(True)
        else:
            self.gauss_view.setEnabled(True)
        self.ax_im = None
        self.update_rgb_image()

        if len(image.shape) > 2:
            val = self.slider.value()
            self.slider.setRange(0, image.shape[0] - 1)
            new_val = int(image.shape[0] / 2)
            self.slider.setValue(new_val)
            if val == new_val:
                self.update_image_view()
        else:
            self.update_image_view()
        if image_update:
            pyplot.xlim(ax_lim)
            pyplot.ylim(ay_lim)

    def update_colormap(self):
        if self.base_image is None:
            return
        self.update_rgb_image()
        self.update_image_view()

    def update_rgb_image(self):
        norm = colors.PowerNorm(gamma=self.settings.power_norm,
                                vmin=self.min_value, vmax=self.max_value)
        if self.gauss_view.isChecked():
            float_image = norm(self.gauss_image)
        else:
            float_image = norm(self.base_image)
        if self.mark_mask.isChecked() and self.settings.mask is not None:
            zero_mask = self.settings.mask == 0
            if self.settings.threshold_type == UPPER:
                float_image[zero_mask] = \
                    (1 - self.settings.mask_overlay) * float_image[zero_mask] + self.settings.mask_overlay
            else:
                float_image[zero_mask] = \
                    (1 - self.settings.mask_overlay) * float_image[zero_mask] + \
                    self.settings.mask_overlay * float_image.min()
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
        self.figure_canvas.draw()

    """def resizeEvent(self, resize_event):
        super(MyCanvas, self).resizeEvent(resize_event)
        # print(self.__class__.__name__, self.size(), resize_event.size(), resize_event.oldSize())
        # self.update_elements_positions()
        self.updateGeometry()"""

    def get_image(self):
        if len(self.base_image.shape) <= 2:
            image_to_show = self.rgb_image
        else:
            image_to_show = self.rgb_image[self.layer_num]
        pyplot.figure(self.my_figure_num)
        return image_to_show, pyplot.xlim(), pyplot.ylim()


class MyDrawCanvas(MyCanvas):
    """
    :type segmentation: np.ndarray
    :type segment: Segment
    """

    def __init__(self, figure_size, settings, info_object, segment, *args):
        super(MyDrawCanvas, self).__init__(figure_size, settings, info_object, *args)
        self.draw_canvas = DrawObject(settings, segment, self.draw_update)
        self.history_list = list()
        self.redo_list = list()

        self.draw_button = QToolButton(self)
        self.draw_button.setToolTip("Draw")
        self.draw_button.setIcon(QIcon(os.path.join(file_folder, "icons", "draw-path.png")))
        self.draw_button.setIconSize(canvas_icon_size)
        self.draw_button.setCheckable(True)
        # self.draw_button.clicked[bool].connect(self.draw_click)
        self.erase_button = QToolButton(self)
        self.erase_button.setToolTip("Erase")
        self.erase_button.setIcon(QIcon(os.path.join(file_folder, "icons", "draw-eraser.png")))
        self.erase_button.setIconSize(canvas_icon_size)
        self.erase_button.setCheckable(True)
        # self.erase_button.clicked[bool].connect(self.erase_click)
        self.show_button = QPushButton("Show", self)
        self.hide_button = QPushButton("Hide", self)
        self.show_button.setCheckable(True)
        self.hide_button.setCheckable(True)
        self.zoom_button.clicked.connect(self.generate_up_button_fun(self.zoom_button))
        self.move_button.clicked.connect(self.generate_up_button_fun(self.move_button))
        self.draw_button.clicked.connect(self.generate_up_button_fun(self.draw_button))
        self.erase_button.clicked.connect(self.generate_up_button_fun(self.erase_button))
        self.show_button.clicked.connect(self.generate_up_button_fun(self.show_button))
        self.hide_button.clicked.connect(self.generate_up_button_fun(self.hide_button))
        self.draw_button.clicked[bool].connect(self.generate_draw_modify(DrawType.draw))
        self.erase_button.clicked[bool].connect(self.generate_draw_modify(DrawType.erase))
        self.show_button.clicked[bool].connect(self.generate_draw_modify(DrawType.force_show))
        self.hide_button.clicked[bool].connect(self.generate_draw_modify(DrawType.force_hide))
        self.clean_button = QPushButton("Clean drawing", self)
        self.clean_button.clicked.connect(self.draw_canvas.clean)
        self.clean_data_button = QPushButton("Clean data", self)
        self.clean_data_button.clicked.connect(self.action_choose)
        self.button_list.extend([self.draw_button, self.erase_button, self.show_button, self.hide_button,
                                 self.clean_button, self.clean_data_button])
        self.segment = segment
        self.protect_button = False
        self.segmentation = None
        self.rgb_segmentation = None
        self.original_rgb_image = None
        self.labeled_rgb_image = None
        self.cursor_val = 0
        self.colormap_checkbox.setChecked(False)
        self.segment.add_segmentation_callback(self.segmentation_changed)
        self.slider.valueChanged[int].connect(self.draw_canvas.set_layer_num)
        self.slider.valueChanged[int].connect(self.settings.change_layer)
        self.figure_canvas.mpl_connect('button_press_event', self.draw_canvas.on_mouse_down)
        self.figure_canvas.mpl_connect('motion_notify_event', self.draw_canvas.on_mouse_move)
        self.figure_canvas.mpl_connect('button_release_event', self.draw_canvas.on_mouse_up)

    def action_choose(self):
        if str(self.clean_data_button.text()) == "Clean data":
            self.remove_noise()
            self.clean_data_button.setText("Restore image")
        else:
            self.restore_original_image()
            self.clean_data_button.setText("Clean data")

    def restore_original_image(self):
        self.settings.image = self.settings.original_image
        self.settings.image_changed_fun()
        self.settings.image_clean_profile = None

    def remove_noise(self):
        image = np.copy(self.settings.image)
        mask = self.segment.get_segmentation()
        full_mask = self.segment.get_full_segmentation()
        if not np.any(np.array(full_mask == 0)):
            return
        noise_mean = np.mean(image[full_mask == 0])
        noise_mask = np.copy(full_mask)
        # noise_std = np.std(image[full_mask == 0])
        # noise_generator = norm(noise_mean, noise_std)
        noise_mask[mask > 0] = 0
        profile = SegmentationProfile("Clean_profile", **self.settings.get_profile_dict())
        self.settings.image_clean_profile = profile
        image[noise_mask > 0] = noise_mean
        # image[noise_mask > 0] = noise_generator.rvs(np.count_nonzero(noise_mask))
        self.settings.image = image
        self.settings.image_changed_fun()

    def draw_update(self, view=True):
        if view:
            self.update_image_view()
        x = int(self.draw_canvas.f_x)
        y = int(self.draw_canvas.f_y)
        segmentation = self.segment.get_segmentation()
        if len(segmentation.shape) == 3:
            val = segmentation[self.layer_num, y, x]
        else:
            val = segmentation[y, x]
        if val == self.cursor_val:
            return
        else:
            self.cursor_val = val
        if val == 0:
            self.info_object.update_info_text("No component")
        else:
            size = self.segment.get_size_array()[val]
            self.info_object.update_info_text("Component: {}, size: {}".format(val, size))

    def generate_up_button_fun(self, button):
        butt_list = [self.zoom_button, self.move_button, self.draw_button, self.erase_button, self.show_button,
                     self.hide_button]
        butt_list.remove(button)

        def fin_fun():
            if self.protect_button:
                return
            self.protect_button = True
            for butt in butt_list:
                if butt.isChecked():
                    butt.click()
            self.protect_button = False

        return fin_fun

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

    def generate_draw_modify(self, t):
        def draw_modify(checked):
            if checked:
                self.draw_canvas.draw_on = True
                self.draw_canvas.value = t.value
            else:
                self.draw_canvas.draw_on = False
        return draw_modify

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

    def set_image(self, image, gauss, image_update=False):
        self.base_image = image
        pyplot.figure(self.my_figure_num)
        ax_lim = pyplot.xlim()
        ay_lim = pyplot.ylim()
        self.max_value = image.max()
        self.min_value = image.min() / float(self.max_value)
        self.gauss_image = gauss
        self.ax_im = None
        self.original_rgb_image = None
        self.draw_canvas.set_image(image)
        self.segment.set_image()
        self.update_rgb_image()
        if len(image.shape) > 2:
            self.slider.setRange(0, image.shape[0] - 1)
            self.slider.setValue(int(image.shape[0] / 2))
        else:
            self.update_image_view()
        if image_update:
            pyplot.xlim(ax_lim)
            pyplot.ylim(ay_lim)

    def update_segmentation_image(self):
        if not self.segment.segmentation_changed:
            return
        self.segmentation = np.copy(self.segment.get_segmentation())
        self.segmentation[self.segmentation > 0] += 2
        self.rgb_segmentation = label_to_rgb(self.segmentation)

    def get_rgb_segmentation_and_mask(self):
        if self.rgb_segmentation.ndim == 2:
            return self.rgb_segmentation, self.segment.get_segmentation()
        else:
            return self.rgb_segmentation[self.layer_num], self.segment.get_segmentation()[self.layer_num]


class DrawObject(object):
    """
    :type settings: Settings
    """
    def __init__(self, settings, segment, update_fun):
        """
        :type settings: Settings
        :type segment: Segment
        :param update_fun:
        """
        self.settings = settings
        self.segment = segment
        self.mouse_down = False
        self.draw_on = False
        self.draw_canvas = None
        self.prev_x = None
        self.prev_y = None
        self.original_rgb_image = None
        self.rgb_image = None
        self.labeled_rgb_image = None
        self.layer_num = 0
        im = [np.arange(5)]
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

    def generate_set_mode(self, t):
        def set_mode():
            self.draw_on = True
            self.value = t.value
        return set_mode

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
        self.rgb_image[pos] = self.original_rgb_image[pos] * (1 - self.settings.overlay) + \
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
        self.segment.draw_counter = 0
        self.update_fun()
        self.settings.metadata_changed()

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
        if self.value > 2:
            return
        if self.mouse_down and self.draw_on and event.xdata is not None and event.ydata is not None:
            f_x, f_y = event.xdata + 0.5, event.ydata + 0.5
            # ix, iy = int(f_x), int(f_y)
            max_dist = max(abs(f_x - self.f_x), abs(f_y - self.f_y))
            rep_num = int(max_dist * 2) + 2
            points = set()
            for fx, fy in zip(np.linspace(f_x, self.f_x, num=rep_num), np.linspace(f_y, self.f_y, num=rep_num)):
                points.add((int(fx), int(fy)))
            points.remove((int(self.f_x), int(self.f_y)))
            for fx, fy in points:
                self.draw_fun((self.layer_num, fy, fx))
            self.f_x = f_x
            self.f_y = f_y
            self.update_fun()
        elif event.xdata is not None and event.ydata is not None:
            self.f_x, self.f_y = int(event.xdata) + 0.5, (event.ydata + 0.5)
            self.update_fun(False)

    def on_mouse_up(self, _):
        self.history.append(self.click_history)
        self.segment.draw_counter = np.count_nonzero(self.draw_canvas)
        self.mouse_down = False
        self.settings.metadata_changed()


class MaskWindow(QDialog):
    """
    :type settings: Settings
    """
    def __init__(self, settings, segment, settings_updated_function):
        super(MaskWindow, self).__init__()
        self.settings = settings
        self.segment = segment
        self.settings_updated_function = settings_updated_function
        main_layout = QVBoxLayout()
        dilate_label = QLabel("Dilate (x,y) radius (in pixels)", self)
        self.dilate_radius = QSpinBox(self)
        self.dilate_radius.setRange(0, 100)
        self.dilate_radius.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.dilate_radius.setValue(settings.mask_dilate_radius)
        self.dilate_radius.setSingleStep(1)
        dilate_layout = QHBoxLayout()
        dilate_layout.addWidget(dilate_label)
        dilate_layout.addWidget(self.dilate_radius)
        main_layout.addLayout(dilate_layout)
        op_layout = QHBoxLayout()
        if len(settings.next_segmentation_settings) == 0:
            self.save_draw = QCheckBox("Save draw", self)
        else:
            self.save_draw = QCheckBox("Add draw", self)
        op_layout.addWidget(self.save_draw)
        self.reset_next = QPushButton("Reset Next")
        self.reset_next.clicked.connect(self.reset_next_fun)
        if len(settings.next_segmentation_settings) == 0:
            self.reset_next.setDisabled(True)
        op_layout.addStretch()
        op_layout.addWidget(self.reset_next)
        main_layout.addLayout(op_layout)
        self.prev_button = QPushButton("Previous mask ({})".format(len(settings.prev_segmentation_settings)), self)
        if len(settings.prev_segmentation_settings) == 0:
            self.prev_button.setDisabled(True)
        self.cancel = QPushButton("Cancel", self)
        self.cancel.clicked.connect(self.close)
        self.next_button = QPushButton("Next mask ({})".format(len(settings.next_segmentation_settings)), self)
        if len(settings.next_segmentation_settings) == 0:
            self.next_button.setText("Next mask (new)")
        self.next_button.clicked.connect(self.next_mask)
        self.prev_button.clicked.connect(self.prev_mask)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.cancel)
        button_layout.addWidget(self.next_button)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def reset_next_fun(self):
        self.settings.next_segmentation_settings = []
        self.next_button.setText("Next mask (new)")
        self.reset_next.setDisabled(True)

    def next_mask(self):
        self.settings.mask_dilate_radius = self.dilate_radius.value()
        self.settings.change_segmentation_mask(self.segment, MaskChange.next_seg, self.save_draw.isChecked())
        self.settings_updated_function()
        self.close()

    def prev_mask(self):
        self.settings.change_segmentation_mask(self.segment, MaskChange.prev_seg, False)
        self.settings_updated_function()
        self.close()


# noinspection PyArgumentList
class MainMenu(QWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings, segment, *args, **kwargs):
        """
        :type settings: Settings
        :type segment: Segment
        :type args: list
        :type kwargs: dict
        """
        super(MainMenu, self).__init__(*args, **kwargs)
        self.settings = settings
        self.segment = segment
        self.settings.add_image_callback(self.set_threshold_range)
        self.settings.add_image_callback(self.set_layer_threshold)
        self.settings.add_change_layer_callback(self.changed_layer)
        self.load_button = QToolButton(self)
        self.load_button.setIcon(QIcon(os.path.join(file_folder, "icons", "document-open.png")))
        self.load_button.setIconSize(QSize(30,30))
        # self.load_button.setStyleSheet("padding: 3px;")
        self.load_button.setToolTip("Open")
        self.load_button.clicked.connect(self.open_file)
        self.save_button = QToolButton(self)
        self.save_button.setIcon(QIcon(os.path.join(file_folder, "icons", "document-save-as.png")))
        self.save_button.setIconSize(QSize(30, 30))
        # self.save_button.setStyleSheet("padding: 3px;")
        self.save_button.setToolTip("Save")
        self.save_button.setDisabled(True)
        self.save_button.clicked.connect(self.save_file)
        self.mask_button = QPushButton("Mask manager", self)
        self.mask_button.setDisabled(True)
        self.mask_button.clicked.connect(self.segmentation_to_mask)
        self.threshold_type = QComboBox(self)
        self.threshold_type.addItem("Upper threshold:")
        self.threshold_type.addItem("Lower threshold:")
        self.threshold_type.currentIndexChanged[str_type].connect(settings.change_threshold_type)
        self.threshold_value = QSpinBox(self)
        self.threshold_value.setMinimumWidth(80)
        self.threshold_value.setRange(0, 100000)
        self.threshold_value.setAlignment(Qt.AlignRight)
        self.threshold_value.setValue(self.settings.threshold)
        self.threshold_value.setSingleStep(500)
        self.threshold_value.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.threshold_value.valueChanged.connect(self.threshold_change)
        self.threshold_timer = QTimer()
        self.threshold_timer.setSingleShot(True)
        self.threshold_timer.timeout.connect(self.threshold_changed)
        self.layer_thr_check = QCheckBox("Layer\nthreshold", self)
        self.layer_thr_check.clicked[bool].connect(self.settings.change_layer_threshold)
        self.minimum_size_lab = QLabel(self)
        self.minimum_size_lab.setText("Minimum object size:")
        self.minimum_size_value = QSpinBox(self)
        self.minimum_size_value.setMinimumWidth(60)
        self.minimum_size_value.setAlignment(Qt.AlignRight)
        self.minimum_size_value.setRange(0, 10 ** 6)
        self.minimum_size_value.setValue(self.settings.minimum_size)
        self.minimum_size_value.valueChanged[int].connect(self.minimum_size_change)
        self.minimum_size_value.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.minimum_size_value.setSingleStep(10)
        self.minimum_size_timer = QTimer()
        self.minimum_size_timer.timeout.connect(self.minimum_size_changed)
        self.minimum_size_timer.setSingleShot(True)
        self.gauss_check = QCheckBox("Use gauss", self)
        self.gauss_check.stateChanged[int].connect(settings.change_gauss)
        self.draw_check = QCheckBox("Use draw\n result", self)
        self.draw_check.stateChanged[int].connect(settings.change_draw_use)
        self.profile_choose = QComboBox(self)
        self.profile_choose.addItem("<no profile>")
        self.profile_choose.addItems(list(sorted(self.settings.get_profile_list())))
        self.profile_choose.currentIndexChanged[str_type].connect(self.profile_changed)
        self.advanced_button = QToolButton(self)  # "Advanced"
        self.advanced_button.setIcon(QIcon(os.path.join(file_folder, "icons", "configure.png")))
        self.advanced_button.setIconSize(QSize(30, 30))
        self.advanced_button.setToolTip("Advanced settings and statistics")
        self.advanced_button.clicked.connect(self.open_advanced)
        self.advanced_window = None

        self.colormap_choose = QComboBox(self)
        self.colormap_choose.addItems(sorted(settings.colormap_list, key=lambda x: x.lower()))
        index = sorted(settings.colormap_list, key=lambda x: x.lower()).index(settings.color_map_name)
        self.colormap_choose.setCurrentIndex(index)
        self.colormap_choose.currentIndexChanged.connect(self.colormap_changed)
        self.settings.add_colormap_list_callback(self.colormap_list_changed)
        self.colormap_protect = False
        # self.setMinimumHeight(50)
        self.update_elements_positions()
        self.one_line = True
        self.mask_window = None
        self.settings.add_profiles_list_callback(self.profile_list_update)
        self.minimum_size_value.valueChanged.connect(self.no_profile)
        self.threshold_value.valueChanged.connect(self.no_profile)
        self.threshold_type.currentIndexChanged.connect(self.no_profile)
        self.layer_thr_check.stateChanged.connect(self.no_profile)
        self.enable_list = [self.save_button, self.mask_button]
        # self.setStyleSheet(self.styleSheet()+";border: 1px solid black")

    def minimum_size_change(self):
        self.minimum_size_timer.stop()
        self.minimum_size_timer.start(reaction_time)

    def minimum_size_changed(self):
        self.settings.change_min_size(self.minimum_size_value.value())

    def threshold_change(self):
        self.threshold_timer.stop()
        self.threshold_timer.start(reaction_time)

    def threshold_changed(self):
        self.settings.change_threshold(self.threshold_value.value())

    def no_profile(self):
        self.profile_choose.setCurrentIndex(0)

    def profile_list_update(self):
        self.profile_choose.clear()
        self.profile_choose.addItem("<no profile>")
        self.profile_choose.addItems(list(sorted(self.settings.get_profile_list())))

    def profile_changed(self, name):
        if name == "<no profile>" or name == "":
            return
        self.settings.change_profile(name)
        self.settings_changed()

    def update_elements_positions(self):
        # m_layout = QVBoxLayout()
        layout = QHBoxLayout()
        second_list = [self.gauss_check, self.draw_check, self.profile_choose,
                       self.colormap_choose]
        # layout.addLayout(pack_layout(self.load_button, self.save_button, self.mask_button))
        layout.addWidget(self.load_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.advanced_button)
        layout.addWidget(self.mask_button)
        layout.addLayout(pack_layout(self.threshold_type, self.threshold_value))
        layout.addWidget(self.layer_thr_check)
        layout.addLayout(pack_layout(self.minimum_size_lab, self.minimum_size_value))
        for el in second_list:
            layout.addWidget(el)
        layout.addStretch()
        # self.setMinimumHeight(50)
        layout.setContentsMargins(0, 0, 0, 0)
        # m_layout.addLayout(layout)
        # info_layout = QHBoxLayout()
        # info_layout.addWidget(QLabel("Test"))
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

    def colormap_changed(self):
        if self.colormap_protect:
            return
        self.settings.change_colormap(str(self.colormap_choose.currentText()))

    def settings_changed(self):
        self.segment.protect = True
        self.minimum_size_value.setValue(self.settings.minimum_size)
        if self.settings.threshold_layer_separate:
            new_threshold = self.settings.threshold_list[self.settings.layer_num]
        else:
            new_threshold = self.settings.threshold
        if new_threshold < self.threshold_value.minimum():
            self.threshold_value.setMinimum(new_threshold - 1)
        if new_threshold > self.threshold_value.maximum():
            self.threshold_value.setMaximum(new_threshold + 1)
        self.threshold_value.setValue(new_threshold)
        self.gauss_check.setChecked(self.settings.use_gauss)
        self.draw_check.setChecked(self.settings.use_draw_result)
        self.layer_thr_check.setChecked(self.settings.threshold_layer_separate)
        if self.settings.threshold_type != UPPER:
            self.threshold_type.setCurrentIndex(
                self.threshold_type.findText("Lower threshold:")
            )
        self.segment.protect = False

    def colormap_list_changed(self):
        self.colormap_protect = True
        text = str(self.colormap_choose.currentText())
        self.colormap_choose.clear()
        self.colormap_choose.addItems(self.settings.colormap_list)
        index = list(self.settings.colormap_list).index(text)
        self.colormap_choose.setCurrentIndex(index)
        self.colormap_protect = False

    def set_threshold_range(self, image):
        val_min = image.min()
        val_max = image.max()
        if self.settings.threshold_layer_separate:
            new_threshold = self.settings.threshold_list[self.settings.layer_num]
        else:
            new_threshold = self.settings.threshold
        self.threshold_value.setRange(val_min, val_max)
        self.threshold_value.setValue(new_threshold)
        diff = val_max - val_min
        if diff > 10000:
            self.threshold_value.setSingleStep(500)
        elif diff > 1000:
            self.threshold_value.setSingleStep(100)
        elif diff > 100:
            self.threshold_value.setSingleStep(20)
        else:
            self.threshold_value.setSingleStep(1)

    def set_layer_threshold(self, _):
        self.layer_thr_check.setChecked(False)

    def changed_layer(self, lay_num):
        self.threshold_value.setValue(lay_num)

    def segmentation_to_mask(self):
        self.mask_window = MaskWindow(self.settings, self.segment, self.settings_changed)
        self.mask_window.exec_()

    def open_file(self):
        dial = QFileDialog(self, "Load data")
        if self.settings.open_directory is not None:
            dial.setDirectory(self.settings.open_directory)
        dial.setFileMode(QFileDialog.ExistingFile)
        filters = ["raw image (*.tiff *.tif *.lsm)", "image with mask (*.tiff *.tif *.lsm *json)",
                   "saved project (*.tgz *.tbz2 *.gz *.bz2)", "Profiles (*.json)"]
        # dial.setFilters(filters)
        dial.setNameFilters(filters)
        if self.settings.open_filter is not None:
            dial.selectNameFilter(self.settings.open_filter)
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            self.settings.open_directory = os.path.dirname(str(file_path))
            selected_filter = str(dial.selectedNameFilter())
            self.settings.open_filter = selected_filter
            logging.debug("open file: {}, filter {}".format(file_path, selected_filter))
            # TODO maybe something better. Now main window have to be parent
            if selected_filter == "raw image (*.tiff *.tif *.lsm)":
                im = tifffile.imread(file_path)
                if im.ndim == 4:
                    choose = MultiChannelFilePreview(im, self.settings)
                    if choose.exec_():
                        index, num = choose.get_result()
                        im = im.take(num, axis=index)
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
                    if image.ndim == 4:
                        choose = MultiChannelFilePreview(image, self.settings)
                        if choose.exec_():
                            index, num = choose.get_result()
                            image = image.take(num, axis=index)
                        else:
                            return
                    org_name = os.path.basename(file_path)
                    mask_dial = QFileDialog(self, "Load mask for {}".format(org_name))
                    filters = ["mask (*.tiff *.tif *.lsm)"]
                    mask_dial.setNameFilters(filters)
                    if mask_dial.exec_():
                        # print(mask_dial.selectedFiles()[0])
                        mask = tifffile.imread(str(mask_dial.selectedFiles()[0]))
                        if mask.ndim == 4:
                            choose = MultiChannelFilePreview(mask, self.settings)
                            if choose.exec_():
                                index, num = choose.get_result()
                                mask = mask.take(num, axis=index)
                            else:
                                return
                        if image.shape != mask.shape:
                            QMessageBox.critical(self, "Wrong shape", "Image and mask has different shapes")
                            return
                        self.settings.add_image(image, file_path, mask)
            elif selected_filter == "saved project (*.tgz *.tbz2 *.gz *.bz2)":
                load_project(file_path, self.settings, self.segment)
                self.settings_changed()
                # self.segment.threshold_updated()
            elif selected_filter == "Profiles (*.json)":
                self.settings.load_profiles(file_path)
            else:
                # noinspection PyCallByClass
                _ = QMessageBox.warning(self, "Load error", "Function do not implemented yet")
                return
            for el in self.enable_list:
                el.setEnabled(True)
            self.settings.advanced_settings_changed()

    def save_file(self):
        dial = QFileDialog(self, "Save data")
        if self.settings.save_directory is not None:
            dial.setDirectory(self.settings.save_directory)
        dial.setFileMode(QFileDialog.AnyFile)
        filters = ["Project (*.tgz *.tbz2 *.gz *.bz2)", "Labeled image (*.tif)", "Mask in tiff (*.tif)",
                   "Mask for itk-snap (*.img)", "Data for chimera (*.cmap)", "Image (*.tiff)", "Profiles (*.json)"]
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setNameFilters(filters)
        default_name = os.path.splitext(os.path.basename(self.settings.file_path))[0]
        dial.selectFile(default_name)
        if self.settings.save_filter is not None:
            dial.selectNameFilter(self.settings.save_filter)
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            selected_filter = str(dial.selectedNameFilter())
            self.settings.save_filter = selected_filter
            self.settings.save_directory = os.path.dirname(file_path)
            if os.path.splitext(file_path)[1] == '':
                ext = re.search(r'\(\*(\.\w+)', selected_filter).group(1)
                file_path += ext
                if os.path.exists(file_path):
                    # noinspection PyCallByClass
                    ret = QMessageBox.warning(self, "File exist", os.path.basename(file_path) +
                                              " already exists.\nDo you want to replace it?",
                                              QMessageBox.No | QMessageBox.Yes)
                    if ret == QMessageBox.Yes:
                        self.save_file()
                        return

            if selected_filter == "Project (*.gz *.bz2)":
                save_to_project(file_path, self.settings, self.segment)

            elif selected_filter == "Labeled image (*.tif)":
                segmentation = self.segment.get_segmentation()
                segmentation[segmentation>0] +=2
                image = np.copy(self.settings.image)
                cmap = matplotlib.cm.get_cmap("gray")
                float_image = image / float(image.max())
                rgb_image = cmap(float_image)
                label_image = sitk.GetArrayFromImage(sitk.LabelToRGB(sitk.GetImageFromArray(segmentation)))
                rgb_image = np.array(rgb_image[..., :3] * 256).astype(np.uint8)
                mask = segmentation > 0
                overlay = self.settings.overlay
                rgb_image[mask] = rgb_image[mask] * (1 - overlay) + label_image[mask] * overlay
                tifffile.imsave(file_path, rgb_image)

            elif selected_filter == "Mask in tiff (*.tif)":
                segmentation = self.segment.get_segmentation()
                tifffile.imsave(file_path, segmentation)
            elif selected_filter == "Mask for itk-snap (*.img)":
                segmentation = sitk.GetImageFromArray(self.segment.get_segmentation())
                sitk.WriteImage(segmentation, file_path)
            elif selected_filter == "Raw Data for chimera (*.cmap)":
                if not np.any(self.segment.get_segmentation()):
                    QMessageBox.warning(self, "No object", "There is no component to export to cmap")
                    return
                save_to_cmap(file_path, self.settings, self.segment, gauss_type=GaussUse.no_gauss,
                             centered_data=False, with_cutting=False)
            elif selected_filter == "Data for chimera (*.cmap)":
                if not np.any(self.segment.get_segmentation()):
                    QMessageBox.warning(self, "No object", "There is no component to export to cmap")
                    return
                ob = CmapSave(file_path, self.settings, self.segment)
                ob.exec_()
                return
                save_to_cmap(file_path, self.settings, self.segment, gauss_type=GaussUse.no_gauss)
            elif selected_filter == "Data for chimera with 2d gauss (*.cmap)":
                if not np.any(self.segment.get_segmentation()):
                    QMessageBox.warning(self, "No object", "There is no component to export to cmap")
                    return
                save_to_cmap(file_path, self.settings, self.segment, gauss_type=GaussUse.gauss_2d)
            elif selected_filter == "Data for chimera with 3d gauss (*.cmap)":
                if not np.any(self.segment.get_segmentation()):
                    QMessageBox.warning(self, "No object", "There is no component to export to cmap")
                    return
                save_to_cmap(file_path, self.settings, self.segment, gauss_type=GaussUse.gauss_3d)
            elif selected_filter == "Image (*.tiff)":
                image = self.settings.image
                tifffile.imsave(file_path, image)
            elif selected_filter == "Profiles (*.json)":
                self.settings.dump_profiles(file_path)
            else:
                # noinspection PyCallByClass
                _ = QMessageBox.critical(self, "Save error", "Option unknown")

    def open_advanced(self):
        if self.advanced_window is not None and self.advanced_window.isVisible():
            self.advanced_window.activateWindow()
            return
        self.advanced_window = AdvancedWindow(self.settings, self.segment)
        self.advanced_window.show()


class InfoMenu(QLabel):
    def __init__(self, settings, segment, parent):
        """
        :type settings: Settings
        :type segment: Segment
        :type parent: QWidget
        """
        super(InfoMenu, self).__init__(parent)
        self.settings = settings
        self.segment = segment
        layout = QHBoxLayout()
        grid_layout = QGridLayout()
        # self.tester = QLabel("TEST", self)
        # layout.addWidget(self.tester)
        self.text_filed = QLabel(self)
        grid_layout.addWidget(self.text_filed, 0, 0)
        self.brightness_field = QLabel(self)
        self.brightness_field.setText("")
        self.brightness_field.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.brightness_field, 0, 1)
        self.info_filed = QLabel(self)
        self.info_filed.setMinimumWidth(100)
        self.info_filed.setAlignment(Qt.AlignRight)
        self.info_filed.setText("No component")
        grid_layout.addWidget(self.info_filed, 0, 2)
        layout.addLayout(grid_layout)
        self.setLayout(layout)
        settings.add_metadata_changed_callback(self.update_text)
        self.update_text()
        self.setMinimumHeight(30)
        layout.setContentsMargins(0, 0, 0, 0)

    def update_text(self):
        spacing = self.settings.spacing
        voxel_size = self.settings.voxel_size
        draw_size = self.segment.draw_counter
        logging.debug("Spacing: {}, Voxel size: {},  Number of changed pixels: {},  ".format(
            spacing, voxel_size, draw_size))
        self.text_filed.setText("Spacing: {}, Voxel size: {},  Number of changed pixels: {}, Gauss radius: {} ".format(
            spacing, voxel_size, draw_size, self.settings.gauss_radius))

    def update_info_text(self, s):
        self.info_filed.setText(s)

    def update_brightness(self, val):
        if val is None:
            self.brightness_field.setText("")
        else:
            self.brightness_field.setText("Pixel brightness: {}".format(val))


def synchronize_zoom(fig1, fig2, sync_checkbox):
    """
    :type fig1: MyCanvas
    :type fig2: MyCanvas
    :type sync_checkbox: QCheckBox
    :return:
    """
    sync_checkbox.stateChanged[int].connect(fig1.sync_zoom)
    sync_checkbox.stateChanged[int].connect(fig2.sync_zoom)
    fig1.sync_fig_num = fig2.my_figure_num
    fig2.sync_fig_num = fig1.my_figure_num


class MainWindow(QMainWindow):
    def __init__(self, title, path_to_open, dev):
        super(MainWindow, self).__init__()
        self.open_path = path_to_open
        print (path_to_open)
        self.setWindowTitle(title)
        self.title = title
        self.settings = Settings(os.path.join(config_folder, "settings.json"))
        self.segment = Segment(self.settings)
        self.main_menu = MainMenu(self.settings, self.segment, self)
        self.info_menu = InfoMenu(self.settings, self.segment, self)

        self.normal_image_canvas = MyCanvas((12, 12), self.settings, self.info_menu, self)
        self.colormap_image_canvas = ColormapCanvas((1, 12), self.settings, self)
        self.segmented_image_canvas = MyDrawCanvas((12, 12), self.settings, self.info_menu, self.segment, self)
        self.segmented_image_canvas.segment.add_segmentation_callback((self.update_object_information,))
        self.normal_image_canvas.update_elements_positions()
        self.segmented_image_canvas.update_elements_positions()
        self.slider_swap = QCheckBox("Synchronize\nsliders", self)
        self.slider_swap.setChecked(True)
        self.sync = SynchronizeSliders(self.normal_image_canvas.slider, self.segmented_image_canvas.slider,
                                       self.slider_swap)
        self.colormap_image_canvas.set_bottom_widget(self.slider_swap)
        self.zoom_sync = QCheckBox("Synchronize\nzoom", self)
        # self.zoom_sync.setDisabled(True)
        synchronize_zoom(self.normal_image_canvas, self.segmented_image_canvas, self.zoom_sync)
        self.colormap_image_canvas.set_top_widget(self.zoom_sync)
        self.colormap_image_canvas.set_layout()

        # noinspection PyArgumentList
        big_font = QFont(QApplication.font())
        big_font.setPointSize(big_font_size)

        self.object_count = QLabel(self)
        self.object_count.setFont(big_font)
        self.object_count.setFixedWidth(150)
        self.object_size_list = QTextEdit(self)
        self.object_size_list.setReadOnly(True)
        self.object_size_list.setFont(big_font)
        self.object_size_list.setMinimumWidth(500)
        self.object_size_list.setMaximumHeight(30)
        # self.object_size_list.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # self.object_size_list_area.setWidget(self.object_size_list)
        # self.object_size_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.object_size_list.setMinimumHeight(200)
        # self.object_size_list.setWordWrap(True)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.settings.add_image_callback((self.set_info, str))

        # self.setGeometry(0, 0,  1400, 720)
        icon = QIcon(os.path.join(file_folder, "icon.png"))
        self.setWindowIcon(icon)
        menu_bar = self.menuBar()
        menu = menu_bar.addMenu("File")

        menu.addAction("Load").triggered.connect(self.main_menu.open_file)
        save = menu.addAction("Save")
        save.setDisabled(True)
        save.triggered.connect(self.main_menu.save_file)
        export = menu.addAction("Export")
        export.setDisabled(True)
        export.triggered.connect(self.export)
        self.main_menu.enable_list.extend([save, export])
        if dev:
            batch = menu.addAction("Batch processing")
            batch.triggered.connect(self.batch_view)
        exit_menu = menu.addAction("Exit")
        exit_menu.triggered.connect(self.close)

        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("Help").triggered.connect(self.help)
        help_menu.addAction("Credits").triggered.connect(self.credits)
        self.credits_widget = None
        self.help_widget = None
        self.batch_widget = None

        self.update_objects_positions()
        self.settings.add_image(tifffile.imread(os.path.join(file_folder, "clean_segment.tiff")), "")

    def batch_view(self):
        if self.batch_widget is not None and self.batch_widget.isVisible():
            self.batch_widget.activateWindow()
            return
        self.batch_widget = BatchWindow(self.settings)
        self.batch_widget.show()

    def help(self):
        self.help_widget = HelpWindow()
        self.help_widget.show()

    def credits(self):
        self.credits_widget = Credits(self)
        self.credits_widget.exec_()

    def set_info(self, image, txt):
        self.statusBar.showMessage("{} {}".format(txt, image.shape))
        self.setWindowTitle("{}: {} {}".format(self.title, os.path.basename(txt), image.shape))

    def export(self):
        dial = QFileDialog(self, "Save data")
        if self.settings.export_directory is not None:
            dial.setDirectory(self.settings.export_directory)
        dial.setFileMode(QFileDialog.AnyFile)
        filters = ["Labeled layer (*.png)", "Clean layer (*.png)", "Only label (*.png)"]
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setNameFilters(filters)
        default_name = os.path.splitext(os.path.basename(self.settings.file_path))[0]
        dial.selectFile(default_name)
        if self.settings.export_filter is not None:
            dial.selectNameFilter(self.settings.export_filter)
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            if os.path.splitext(file_path)[1] != ".png":
                file_path += ".png"
                if os.path.exists(file_path):
                    # noinspection PyCallByClass
                    ret = QMessageBox.warning(self, "File exist", os.path.basename(file_path) +
                                              " already exists.\nDo you want to replace it?",
                                              QMessageBox.No, QMessageBox.Yes)
                    if ret == QMessageBox.No:
                        self.export()
                        return
            selected_filter = str(dial.selectedNameFilter())
            self.settings.export_filter = selected_filter
            self.settings.export_directory = os.path.dirname(file_path)
            if selected_filter == "Labeled layer (*.png)":
                ie = ImageExporter(self.segmented_image_canvas, file_path, selected_filter, self)
                ie.exec_()
            elif selected_filter == "Clean layer (*.png)":
                ie = ImageExporter(self.normal_image_canvas, file_path, selected_filter, self)
                ie.exec_()
            elif selected_filter == "Only label (*.png)":
                seg, mask = self.segmented_image_canvas.get_rgb_segmentation_and_mask()
                seg = np.dstack((seg, np.zeros(seg.shape[:-1], dtype=np.uint8)))
                seg[..., 3][mask > 0] = 255
                ie = ImageExporter(self.segmented_image_canvas, file_path, selected_filter, self,
                                   image=seg)
                ie.exec_()
            else:
                _ = QMessageBox.critical(self, "Save error", "Option unknown")

    def showEvent(self, _):
        try:
            if self.open_path is not None:
                if os.path.splitext(self.open_path)[1] in ['.bz2', ".tbz2", ".gz", "tgz"]:
                    load_project(self.open_path, self.settings, self.segment)
                elif os.path.splitext(self.open_path)[1] in ['.tif', '.tiff', '*.lsm']:
                    im = tifffile.imread(self.open_path)
                    if im.ndim < 4:
                        self.settings.add_image(im, self.open_path)
                    else:
                        return
                for el in self.main_menu.enable_list:
                    el.setEnabled(True)
                self.main_menu.settings_changed()
        except Exception as e:
            logging.warning(e.message)

    def update_objects_positions(self):
        widget = QWidget()
        main_layout = QVBoxLayout()
        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.main_menu)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(menu_layout)
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.info_menu)
        info_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(info_layout)
        image_layout = QHBoxLayout()
        image_layout.setSpacing(0)
        image_layout.addWidget(self.normal_image_canvas)
        image_layout.addWidget(self.colormap_image_canvas)
        image_layout.addWidget(self.segmented_image_canvas)
        main_layout.addLayout(image_layout)
        main_layout.addSpacing(5)
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.object_count)
        info_layout.addWidget(self.object_size_list)
        main_layout.addLayout(info_layout)
        main_layout.addStretch()
        main_layout.setSpacing(0)
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
        self.slider_swap.move(col_pos.x() + 5,
                              col_pos.y() + self.colormap_image_canvas.height() - 35)
        # self.infoText.move()

        norm_pos = self.normal_image_canvas.pos()
        self.object_count.move(norm_pos.x(),
                               norm_pos.y() + self.normal_image_canvas.height() + 20)
        self.object_size_list.move(self.object_count.pos().x() + 150, self.object_count.pos().y())

    def update_object_information(self, info_array):
        """:type info_array: np.ndarray"""
        self.object_count.setText("Object num: {0}".format(str(info_array.size)))
        self.object_size_list.setText("Objects size: {}".format(list(info_array)))

    def closeEvent(self, event):
        logging.debug("Close: {}".format(os.path.join(config_folder, "settings.json")))
        if self.batch_widget is not None and self.batch_widget.isVisible():
            if self.batch_widget.is_working():
                ret = QMessageBox.warning(self, "Batch work", "Batch work is not finished. "
                                                              "Would you like to terminate it?",
                                          QMessageBox.No | QMessageBox.Yes)
                if ret == QMessageBox.Yes:
                    self.batch_widget.terminate()
                    self.batch_widget.close()
                else:
                    event.ignore()
                    return
            else:
                self.batch_widget.close()
        self.settings.dump(os.path.join(config_folder, "settings.json"))
        if self.main_menu.advanced_window is not None and self.main_menu.advanced_window.isVisible():
            self.main_menu.advanced_window.close()


class ImageExporter(QDialog):
    interpolation_dict = {"None": Image.NEAREST, "Bilinear": Image.BILINEAR,
                          "Bicubic": Image.BICUBIC,
                          "Lanczos": Image.LANCZOS}  # "Box": Image.BOX, "Hamming": Image.HAMMING,

    def __init__(self, canvas, file_path, filter_name, parent, image=None):
        print(filter_name)
        super(ImageExporter, self).__init__(parent)
        self.image = image
        self.keep_ratio = QCheckBox("Keep oryginal ratio", self)
        self.keep_ratio.setChecked(True)
        self.scale_x = QDoubleSpinBox(self)
        self.scale_x.setSingleStep(1)
        self.scale_x.setRange(0, 100)
        self.scale_x.setValue(1)
        self.scale_x.valueChanged[float].connect(self.scale_x_changed)
        self.scale_x.setDecimals(3)
        self.scale_y = QDoubleSpinBox(self)
        self.scale_y.setSingleStep(1)
        self.scale_y.setRange(0, 100)
        self.scale_y.setValue(1)
        self.scale_y.valueChanged[float].connect(self.scale_y_changed)
        self.scale_y.setDecimals(3)

        self.size_x = QSpinBox(self)
        self.size_x.setSingleStep(1)
        self.size_x.setRange(0, 10000)
        self.size_x.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.size_x.valueChanged[int].connect(self.size_x_changed)
        self.size_y = QSpinBox(self)
        self.size_y.setSingleStep(1)
        self.size_y.setRange(0, 10000)
        self.size_y.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.size_y.valueChanged[int].connect(self.size_y_changed)
        self.x_change = False
        self.y_change = False

        self.canvas = canvas
        im, ax_size, ay_size = canvas.get_image()
        # print(ax_size, ay_size)
        # self.im_shape = np.array([im.shape[1], im.shape[0]], dtype=np.uint32)
        self.ax_size = (ax_size[0] + 0.5, ax_size[1] + 0.5)
        self.ay_size = (ay_size[1] + 0.5, ay_size[0] + 0.5)
        self.im_shape = int(ax_size[1] - ax_size[0]), int(ay_size[0] - ay_size[1])
        self.path = file_path
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Chosen filter: {}".format(filter_name)))
        path_label = QLabel(file_path)
        path_label.setWordWrap(True)
        layout.addWidget(path_label)
        settings_layout = QGridLayout()
        settings_layout.addWidget(self.keep_ratio, 0, 1)
        settings_layout.addWidget(QLabel("Image scale"), 1, 0)
        image_scale_layout = QHBoxLayout()
        image_scale_layout.addWidget(self.scale_x)
        image_scale_layout.addWidget(self.scale_y)
        settings_layout.addLayout(image_scale_layout, 1, 1)
        settings_layout.addWidget(QLabel("Image size"), 2, 0)
        image_size_layout = QHBoxLayout()
        image_size_layout.addWidget(self.size_x)
        image_size_layout.addWidget(self.size_y)
        settings_layout.addLayout(image_size_layout, 2, 1)

        layout.addLayout(settings_layout)
        image_interpolation_layout = QHBoxLayout()
        image_interpolation_layout.addWidget(QLabel("Interpolation type"))
        self.interp_type = QComboBox(self)
        self.interp_type.addItems(list(self.interpolation_dict.keys()))
        find = list(self.interpolation_dict.keys()).index("None")
        if find != -1:
            self.interp_type.setCurrentIndex(find)
        image_interpolation_layout.addWidget(self.interp_type)
        layout.addLayout(image_interpolation_layout)

        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_image)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.close)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def scale_x_changed(self, val):
        if self.keep_ratio.isChecked():
            self.scale_y.setValue(val)
        if self.x_change:
            return
        self.x_change = True
        self.size_x.setValue(self.im_shape[0] * val)
        if val == 0:
            self.save_button.setDisabled(True)
        else:
            self.save_button.setEnabled(True)
        self.x_change = False

    def scale_y_changed(self, val):
        if self.keep_ratio.isChecked():
            self.scale_x.setValue(val)
        if self.y_change:
            return
        self.y_change = True
        self.size_y.setValue(self.im_shape[1] * val)
        if val == 0:
            self.save_button.setDisabled(True)
        else:
            self.save_button.setEnabled(True)
        self.y_change = False

    def size_x_changed(self, val):
        if self.x_change:
            return
        self.x_change = True
        self.scale_x.setValue(val / self.im_shape[0])
        self.x_change = False

    def size_y_changed(self, val):
        if self.y_change:
            return
        self.y_change = True
        self.scale_y.setValue(val / self.im_shape[1])
        self.y_change = False

    def showEvent(self, _):
        self.size_x.setValue(self.im_shape[0])
        self.size_y.setValue(self.im_shape[1])

    def save_image(self):
        if self.image is None:
            np_im, _, _ = self.canvas.get_image()
        else:
            np_im = self.image
        im = Image.fromarray(np_im)
        x_scale = self.scale_x.value()
        y_scale = self.scale_y.value()
        inter_type = self.interpolation_dict[str(self.interp_type.currentText())]
        im2 = im.resize((int(np_im.shape[1] * x_scale), int(np_im.shape[0] * y_scale)), inter_type)
        im2.crop((int(self.ax_size[0] * x_scale), int(self.ay_size[0] * y_scale),
                  int(self.ax_size[1] * x_scale), int(self.ay_size[1] * y_scale))).save(self.path)
        self.accept()


class MultiChannelFilePreview(QDialog):
    def __init__(self, image, settings):
        """
        :type image: np.ndarray
        :type settings: Settings
        """
        QDialog.__init__(self)
        self.image = image
        index = list(image.shape).index(min(image.shape))
        self.preview = MyCanvas((5, 5), settings, None, self, False)
        self.preview.update_elements_positions()
        self.preview.mark_mask.setDisabled(True)
        self.channel_num = QSpinBox(self)
        self.channel_num.setRange(0, image.shape[index] - 1)
        self.channel_pos = QSpinBox(self)
        self.channel_pos.setRange(0, image.ndim-1)
        self.channel_pos.setValue(index)
        self.channel_num.valueChanged.connect(self.set_image)
        self.channel_pos.valueChanged.connect(self.change_channel_pos)
        accept_butt = QPushButton("Open", self)
        discard_butt = QPushButton("Cancel", self)
        discard_butt.clicked.connect(self.close)
        accept_butt.clicked.connect(self.accept)
        layout = QVBoxLayout()
        layout.addWidget(self.preview)
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel position"))
        channel_layout.addWidget(self.channel_pos)
        channel_layout.addStretch()
        channel_layout.addWidget(QLabel("Channel num"))
        channel_layout.addWidget(self.channel_num)
        layout.addLayout(channel_layout)
        button_layout = QHBoxLayout()
        button_layout.addWidget(discard_butt)
        button_layout.addStretch()
        button_layout.addWidget(accept_butt)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.set_image()

    def set_image(self):
        im = self.image.take(self.channel_num.value(), axis=self.channel_pos.value())
        # print(im.shape)
        self.preview.set_image(im, None)

    def change_channel_pos(self, val):
        self.channel_num.setRange(0, self.image.shape[val]-1)
        self.set_image()

    def get_result(self):
        return self.channel_pos.value(),  self.channel_num.value()


class Credits(QDialog):
    def __init__(self, parent):
        super(Credits, self).__init__(parent)
        layout = QVBoxLayout()
        label = QLabel(self)
        close = QPushButton("Close", self)
        close.clicked.connect(self.close)
        layout.addWidget(label)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close)
        layout.addLayout(button_layout)
        self.setLayout(layout)
        author = "<big>Grzegorz Bokota</big><sup>1,2</sup> " \
                 "(<a href=\"g.bokota@cent.uw.edu.pl\">g.bokota@cent.uw.edu.pl</a>)<br>" \
                 "<big>Michał Kadlof</big><sup>1,3</sup> " \
                 "(<a href=\"m.kadlof@cent.uw.edu.pl\">m.kadlof@cent.uw.edu.pl</a>)<br>" \
                 "<big>Dariusz Plewczynski</big><sup>1</sup> (<a href=\"d.plewczynski@cent.uw.edu.pl\">" \
                 "d.plewczynski@cent.uw.edu.pl </a>)<br><br>" \
                 "<sup>1</sup> Laboratory of functional and structural genomics, <i>Center of New Technologies</i>, " \
                 "University of Warsaw " \
                 "(<a href=\"http://nucleus3d.cent.uw.edu.pl/\">nucleus3d.cent.uw.edu.pl</a>) <br>" \
                 "<sup>2</sup> <i>Faculty of Mathematics, Informatics and Mechanics</i>, " \
                 "University of Warsaw (<a href=\"http://www.mimuw.edu.pl/\">mimuw.edu.pl</a>) <br>" \
                 "<sup>3</sup> <i>Faculty of Physics</i>, " \
                 "University of Warsaw (<a href=\"http://www.fuw.edu.pl/\">fuw.edu.pl</a>)" \

        program = "<big><strong>PartSeg</strong></big> <br> program for segmentation connect component of threshold" \
                  " selected regions <br>" \
                  "<u>Version 0.9 beta (under development)</u>"
        separator = "<br><hr><br>"

        licenses = "LGPLv3 for Oxygen icons <a href=\"http://www.kde.org/\">http://www.kde.org/</a><br>" \
                   "GPL for PyQt project"
        text = program + separator + author + separator + licenses
        label.setText(text)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setWordWrap(True)

    def showEvent(self, _):
        print("Credits show")

    def closeEvent(self, _):
        print("Credits close")


class CmapSave(QDialog):
    """
    :type settings: Settings
    """
    def __init__(self, file_path, settings, segment):
        super(CmapSave, self).__init__()
        self.settings = settings
        self.segment = segment
        self.file_path = file_path
        path_label = QLabel("Save path: <i>{}</i>".format(file_path))
        path_label.setWordWrap(True)
        self.gauss_type = QComboBox(self)
        self.gauss_type.addItems(["No gauss", "2d gauss", "2d + 3d gauss"])
        self.center_data = QCheckBox(self)
        self.center_data.setChecked(True)
        self.with_statistics = QCheckBox(self)
        self.with_statistics.setChecked(True)
        self.rotation_axis = QComboBox(self)
        self.rotation_axis.addItems(["None", "x", "y", "z"])
        self.cut_data = QCheckBox(self)
        self.cut_data.setChecked(True)
        grid = QGridLayout()
        grid.addWidget(QLabel("Gauss type"), 0, 0)
        grid.addWidget(self.gauss_type, 0, 1)
        grid.addWidget(QLabel("Center data"), 1, 0)
        grid.addWidget(self.center_data, 1, 1)
        grid.addWidget(QLabel("With statistics"), 2, 0)
        grid.addWidget(self.with_statistics, 2, 1)
        grid.addWidget(QLabel("Rotation axis"), 3, 0)
        grid.addWidget(self.rotation_axis, 3, 1)
        grid.addWidget(QLabel("Cut obsolete area"), 4, 0)
        grid.addWidget(self.cut_data, 4, 1)

        close = QPushButton("Cancel")
        close.clicked.connect(self.close)
        save = QPushButton("Save")
        save.clicked.connect(self.save)

        button_layout  = QHBoxLayout()
        button_layout.addWidget(close)
        button_layout.addStretch()
        button_layout.addWidget(save)

        layout = QVBoxLayout()
        layout.addWidget(path_label)
        layout.addLayout(grid)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def save(self):
        options = {"No gauss": GaussUse.no_gauss, "2d gauss": GaussUse.gauss_2d, "2d + 3d gauss": GaussUse.gauss_3d}
        save_to_cmap(self.file_path, self.settings, self.segment, options[str(self.gauss_type.currentText())],
                     self.with_statistics.isChecked(), self.center_data.isChecked(),
                     rotate=str(self.rotation_axis.currentText()), with_cutting=self.cut_data.isChecked())
        self.accept()


class HelpWindow(QDockWidget):
    def __init__(self, parent=None):
        super(HelpWindow, self).__init__(parent)
        self.help_engine = QHelpEngine(os.path.join(file_folder, "help", "PartSeg.qhc"))
        self.help_engine.setupData()
        self.menu_widget = QTabWidget(self)
        self.menu_widget.addTab(self.help_engine.contentWidget(), "Contents")
        self.menu_widget.addTab(self.help_engine.indexWidget(), "Index")
        self.browser = HelpBrowser(self, self.help_engine)
        self.help_engine.contentWidget().linkActivated.connect(self.browser.setSource)
        self.help_engine.indexWidget().linkActivated.connect(self.browser.setSource)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.insertWidget(0, self.menu_widget)
        self.splitter.insertWidget(1, self.browser)
        self.setWidget(self.splitter)


class HelpBrowser(QTextBrowser):
    def __init__(self, parent, engine):
        super(HelpBrowser, self).__init__(parent)
        self.help_engine = engine

    def loadResource(self, p_int, name):
        if name.scheme() == "qthelp":
            return QVariant(self.help_engine.fileData(name))
        else:
            return QTextBrowser.loadResource(self, p_int, name)


