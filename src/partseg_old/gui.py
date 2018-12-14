# coding=utf-8
from __future__ import print_function, division

import json
import os.path
import re

import SimpleITK as sitk
import matplotlib.colors as colors
import numpy as np
import tifffile
from PIL import Image
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QKeyEvent
from matplotlib import pyplot

from partseg_old.advanced_window import AdvancedWindow
from partseg_old.backend import Settings, Segment, UPPER, MaskChange
from partseg_old.image_view import MyCanvas
from partseg_old.interpolate_dialog import InterpolateDialog
from partseg_old.io_functions import save_to_cmap, save_to_project, load_project, GaussUse, save_to_xyz
from partseg_utils.global_settings import static_file_folder, config_folder
from partseg_old.qt_import import *

__author__ = "Grzegorz Bokota"


reaction_time = 500

logging.debug(config_folder)

if not os.path.isdir(config_folder):
    os.makedirs(config_folder)


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
        self.timer.start(reaction_time)
        norm = colors.PowerNorm(gamma=self.settings.power_norm, vmin=self.val_min, vmax=self.val_max)
        fig = pyplot.figure(self.my_figure_num)
        pyplot.clf()
        ax = fig.add_axes([0.01, 0.01, 0.25, 0.98])
        matplotlib.colorbar.ColorbarBase(ax, cmap=self.settings.color_map, norm=norm, orientation='vertical')
        fig.canvas.draw()
        # self.update_colormap()
        """if abs(self.starting_value - val) > 500:
            self.run_update(True)
            self.starting_value = val
            self.timer.stop()
        else:
            if not self.timer.isActive():
                self.timer.start(2*reaction_time)"""

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
        if not self.settings.normalize_range[2]:
            norm = colors.PowerNorm(gamma=self.settings.power_norm, vmin=self.val_min, vmax=self.val_max)
        else:
            norm = colors.PowerNorm(gamma=self.settings.power_norm,
                                    vmin=self.settings.normalize_range[0],
                                    vmax=self.settings.normalize_range[1])
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


class MaskWindow(QDialog):
    """
    :type settings: Settings
    """
    def __init__(self, settings, segment, settings_updated_function):
        """

        :type settings: Settings
        """
        super(MaskWindow, self).__init__()
        self.setWindowTitle("Mask manager")
        self.settings = settings
        self.segment = segment
        self.settings_updated_function = settings_updated_function
        main_layout = QVBoxLayout()
        dilate_label = QLabel("Dilate (x,y) radius (in pixels)", self)
        self.dilate_radius = QSpinBox(self)
        self.dilate_radius.setRange(-100, 100)
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
        self.fill_holes = QCheckBox("Fill holes", self)
        self.fill_holes.setToolTip("Fill holes that are not connected in 3d")
        op_layout.addWidget(self.fill_holes)
        self.fill_holes_in_2d = QCheckBox("Fill holes in 2d")
        self.fill_holes_in_2d.setToolTip("Fill holes thar are not connected to border on each slice separately")
        op_layout.addWidget(self.fill_holes_in_2d)
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
        self.settings.change_segmentation_mask(self.segment, MaskChange.next_seg, self.save_draw.isChecked(),
                                               self.fill_holes.isChecked(), self.fill_holes_in_2d.isChecked())
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
        self.load_button.setIcon(QIcon(os.path.join(static_file_folder, "icons", "document-open.png")))
        self.load_button.setIconSize(QSize(30,30))
        # self.load_button.setStyleSheet("padding: 3px;")
        self.load_button.setToolTip("Open")
        self.load_button.clicked.connect(self.open_file)
        self.save_button = QToolButton(self)
        self.save_button.setIcon(QIcon(os.path.join(static_file_folder, "icons", "document-save-as.png")))
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
        self.threshold_type.setCurrentIndex(0 if settings.threshold_type == UPPER else 1)
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
        self.gauss_check.setChecked(self.settings.use_gauss)
        self.gauss_check.stateChanged[int].connect(settings.change_gauss)
        self.draw_check = QCheckBox("Use draw\n result", self)
        self.draw_check.stateChanged[int].connect(settings.change_draw_use)
        self.profile_choose = QComboBox(self)
        self.profile_choose.addItem("<no profile>")
        self.profile_choose.addItems(list(sorted(self.settings.get_profile_list())))
        self.profile_choose.currentIndexChanged[str_type].connect(self.profile_changed)
        self.advanced_button = QToolButton(self)  # "Advanced"
        self.advanced_button.setIcon(QIcon(os.path.join(static_file_folder, "icons", "configure.png")))
        self.advanced_button.setIconSize(QSize(30, 30))
        self.advanced_button.setToolTip("Advanced settings and statistics")
        self.advanced_button.clicked.connect(self.open_advanced)
        self.advanced_window = None

        self.colormap_choose = QComboBox(self)
        self.colormap_choose.setToolTip("Select colormap")
        self.colormap_choose.addItems(sorted(settings.colormap_list, key=lambda x: x.lower()))
        index = sorted(settings.colormap_list, key=lambda x: x.lower()).index(settings.color_map_name)
        self.colormap_choose.setCurrentIndex(index)
        self.colormap_choose.currentIndexChanged.connect(self.colormap_changed)
        self.settings.add_colormap_list_callback(self.colormap_list_changed)
        self.colormap_protect = False
        self.interpolate = QPushButton("Interpolate", self)
        self.interpolate.clicked.connect(self.interpolate_exec)
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

    def keyPressEvent(self, event: QKeyEvent):
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_O:
                self.open_file()
            if event.key() == Qt.Key_S:
                self.save_file()

    def interpolate_exec(self):
        dialog = InterpolateDialog(self.settings.spacing)
        if dialog.exec():
            self.settings.rescale_image(dialog.get_zoom_factor())

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
        layout.addWidget(self.interpolate)
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

    def read_image(self, file_path):
        im = tifffile.imread(file_path)
        im = im.squeeze()
        if im.ndim == 4 or im.shape[-1] < 5:
            choose = MultiChannelFilePreview(im, self.settings)
            if choose.exec_():
                index, num = choose.get_result()
                im = im.take(num, axis=index)
                return im
            else:
                return
        return im

    def open_file(self):
        dial = QFileDialog(self, "Load data")
        if self.settings.open_directory is not None:
            dial.setDirectory(self.settings.open_directory)
        dial.setFileMode(QFileDialog.ExistingFile)
        filters = ["raw image (*.tiff *.tif *.lsm)", "image with mask (*.tiff *.tif *.lsm)",
                   "mask to image (*.tiff *.tif *.lsm)", "image with current mask (*.tiff *.tif *.lsm)",
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
                im = self.read_image(file_path)
                if im is None:
                    return
                self.settings.add_image(im, file_path)
            elif selected_filter == "mask to image (*.tiff *.tif *.lsm)":
                im = self.read_image(file_path)
                if im is None:
                    return
                self.settings.add_image(self.settings.image, self.settings.file_path, im)
            elif selected_filter == "image with current mask (*.tiff *.tif *.lsm)":
                im = self.read_image(file_path)
                if im is None:
                    return
                self.settings.add_image(im, file_path, self.settings.mask)
            elif selected_filter == "image with mask (*.tiff *.tif *.lsm)":
                extension = os.path.splitext(file_path)
                if extension == ".json":
                    with open(file_path) as ff:
                        info_dict = json.load(ff)
                    image = tifffile.imread(info_dict["image"])
                    mask = tifffile.imread(info_dict["mask"])
                    self.settings.add_image(image, file_path, mask)
                else:
                    image = self.read_image(file_path)
                    if image is None:
                        return
                    org_name = os.path.basename(file_path)
                    mask_dial = QFileDialog(self, "Load mask for {}".format(org_name))
                    filters = ["mask (*.tiff *.tif *.lsm)"]
                    mask_dial.setNameFilters(filters)
                    if mask_dial.exec_():
                        mask = self.read_image(file_path)
                        if mask is None:
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
                   "Mask for itk-snap (*.img)", "Data for chimera (*.cmap)", "Image (*.tiff)", "Profiles (*.json)",
                   "Segmented data in xyz (*.xyz)"]
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
                    if ret == QMessageBox.No:
                        self.save_file()
                        return

            if selected_filter == "Project (*.tgz *.tbz2 *.gz *.bz2)":
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
                segmentation = np.array(segmentation > 0).astype(np.uint8)
                tifffile.imsave(file_path, segmentation)
            elif selected_filter == "Mask for itk-snap (*.img)":
                segmentation = sitk.GetImageFromArray(self.segment.get_segmentation())
                sitk.WriteImage(segmentation, file_path)
            elif selected_filter == "Data for chimera (*.cmap)":
                if not np.any(self.segment.get_segmentation()):
                    QMessageBox.warning(self, "No object", "There is no component to export to cmap")
                    return
                ob = CmapSave(file_path, self.settings, self.segment)
                ob.exec_()
            elif selected_filter == "Image (*.tiff)":
                image = self.settings.image
                tifffile.imsave(file_path, image)
            elif selected_filter == "Profiles (*.json)":
                self.settings.dump_profiles(file_path)
            elif selected_filter == "Segmented data in xyz (*.xyz)":
                save_to_xyz(file_path, self.settings, self.segment)
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
        voxel_size = self.settings.voxel_size
        draw_size = self.segment.draw_counter
        logging.debug("Voxel size: {},  Number of changed pixels: {},  ".format(
            voxel_size, draw_size))
        self.text_filed.setText("Voxel size: {},  Number of changed pixels: {}, Gauss radius: {} ".format(
            voxel_size, draw_size, self.settings.gauss_radius))

    def update_info_text(self, s):
        self.info_filed.setText(s)

    def update_brightness(self, val, coords=None):
        if val is None:
            self.brightness_field.setText("")
        else:
            self.brightness_field.setText("Pixel {} value: {}".format(coords, val))


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




class ImageExporter(QDialog):
    interpolation_dict = {"None": Image.NEAREST, "Bilinear": Image.BILINEAR,
                          "Bicubic": Image.BICUBIC,
                          "Lanczos": Image.LANCZOS}  # "Box": Image.BOX, "Hamming": Image.HAMMING,

    def __init__(self, canvas, file_path, filter_name, parent, image=None):
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
        logging.debug("Credits show")

    def closeEvent(self, _):
        logging.debug("Credits close")


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
        self.help_engine = QHelpEngine(os.path.join(os.path.dirname(os.path.dirname(static_file_folder)),
                                                    "help", "PartSeg.qhc"))
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


