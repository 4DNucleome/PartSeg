from __future__ import division, print_function

import collections
import os
from enum import Enum
from math import log
from typing import Type, List, Union, Callable, Optional
from PartSegData import icons_dir

import numpy as np
from qtpy import QtGui, QtCore
from qtpy.QtCore import QRect, QSize, QObject, Signal, QPoint, Qt, QEvent, Slot
from qtpy.QtGui import QWheelEvent, QPainter, QPen, QColor, QPalette, QPixmap, QImage, QIcon, QResizeEvent, QMouseEvent
from qtpy.QtWidgets import QLabel, QGridLayout
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QAction,
    QScrollBar,
    QCheckBox,
    QComboBox,
)

from PartSeg.common_gui.numpy_qimage import NumpyQImage
from PartSegCore.class_generator import enum_register
from PartSegCore.image_operations import apply_filter, NoiseFilterType
from PartSegCore.color_image import color_image_fun, add_labels
from PartSegCore.color_image.color_image_base import color_maps
from PartSegCore.colors import default_colors
from ..common_backend.base_settings import BaseSettings, ViewSettings
from PartSegImage import Image
from .channel_control import ColorComboBoxGroup, ChannelProperty

canvas_icon_size = QSize(20, 20)
step = 1.01
max_step = log(1.2, step)


class LabelEnum(Enum):
    Not_show = 0
    Show_results = 1
    Show_selected = 2

    def __str__(self):
        if self.value == 0:
            return "Don't show"
        return self.name.replace("_", " ")


enum_register.register_class(LabelEnum)


class ImageShowState(QObject):
    """Object for storing state used when presenting it in :class:`.ImageView`"""

    parameter_changed = Signal()  # signal informing that some of image presenting parameters
    # changed and image need to be refreshed

    def __init__(self, settings: ViewSettings, name: str):
        if len(name) == 0:
            raise ValueError("Name string should be not empty")
        super().__init__()
        self.name = name
        self.settings = settings
        self.zoom = False
        self.move = False
        self.opacity = settings.get_from_profile(f"{name}.image_state.opacity", 1.0)
        self.show_label = settings.get_from_profile(f"{name}.image_state.show_label", LabelEnum.Show_results)
        self.only_borders = settings.get_from_profile(f"{name}.image_state.only_border", True)
        self.borders_thick = settings.get_from_profile(f"{name}.image_state.border_thick", 1)

    def set_zoom(self, val):
        self.zoom = val

    def set_move(self, val):
        self.move = val

    def set_borders(self, val: bool):
        """decide if draw only component 2D borders, or whole area"""
        if self.only_borders != val:
            self.settings.set_in_profile(f"{self.name}.image_state.only_border", val)
            self.only_borders = val
            self.parameter_changed.emit()

    def set_borders_thick(self, val: int):
        """If draw only 2D borders of component then set thickness of line used for it"""
        if val != self.borders_thick:
            self.settings.set_in_profile(f"{self.name}.image_state.border_thick", val)
            self.borders_thick = val
            self.parameter_changed.emit()

    def set_opacity(self, val: float):
        """Set opacity of component labels"""
        if self.opacity != val:
            self.settings.set_in_profile(f"{self.name}.image_state.opacity", val)
            self.opacity = val
            self.parameter_changed.emit()

    def components_change(self):
        if self.show_label == LabelEnum.Show_selected:
            self.parameter_changed.emit()

    def set_show_label(self, val: LabelEnum):
        if self.show_label != val:
            self.settings.set_in_profile(f"{self.name}.image_state.show_label", val)
            self.show_label = val
            self.parameter_changed.emit()


class ImageCanvas(QLabel):
    """Canvas for painting image"""

    zoom_mark = Signal(QPoint, QPoint, QSize)  # Signal emitted on end of marking zoom area.
    # Contains two oposit corners of rectangle and current size of canvas
    position_signal = Signal([QPoint, QSize], [QPoint])
    click_signal = Signal([QPoint, QSize], [QPoint])
    leave_signal = Signal()  # mouse left Canvas area

    def __init__(self, local_settings: ImageShowState):
        """
        :type local_settings: ImageShowState
        :param local_settings:
        """
        super().__init__()
        self.scale_factor = 1
        self.local_settings = local_settings
        self.point = None
        self.point2 = None
        self.image = None
        self.image_size = QSize(1, 1)
        self.image_ratio = 1
        self.setMouseTracking(True)
        self.my_pixmap = None

    def set_image(self, im: np.ndarray):
        """set image which will be shown. This function is called from
         :class: `ImageView` when changing image or layer"""
        self.image = im
        height, width, _ = im.shape
        self.image_size = QSize(width, height)
        self.image_ratio = float(width) / float(height)
        self.paint_image()

    def paint_image(self):
        if self.image is None:
            return
        im = self.image
        width, height = self.image_size.width(), self.image_size.height()
        im2 = QImage(im.data, width, height, im.dtype.itemsize * width * 3, QImage.Format_RGB888)
        self.my_pixmap = QPixmap.fromImage(im2)
        self.repaint()

    def leaveEvent(self, a0: QEvent):
        self.point = None
        self.point2 = None
        self.leave_signal.emit()

    def _calculate_real_position(self, pos: QPoint):
        x = int(pos.x() / (self.width() / self.image_size.width()))
        y = int(pos.y() / (self.height() / self.image_size.height()))
        return QPoint(x, y)

    def mousePressEvent(self, event: QMouseEvent):
        super().mousePressEvent(event)
        if self.local_settings.zoom:
            self.point = event.pos()
        elif not self.local_settings.move:
            self.click_signal.emit(event.pos(), self.size())
            self.click_signal[QPoint].emit(self._calculate_real_position(event.pos()))

    def mouseMoveEvent(self, event: QMouseEvent):
        super().mouseMoveEvent(event)
        if self.local_settings.zoom and self.point is not None:
            self.point2 = event.pos()
            self.update()
        self.position_signal.emit(event.pos(), self.size())
        self.position_signal[QPoint].emit(self._calculate_real_position(event.pos()))

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.local_settings.zoom and self.point is not None and self.point2 is not None:
            diff = self.point2 - self.point
            if abs(diff.x()) and abs(diff.y()):
                self.zoom_mark.emit(self.point, self.point2, self.size())
            self.point2 = None
            self.point = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.my_pixmap is not None:
            painter.drawPixmap(self.rect(), self.my_pixmap)
        if not self.local_settings.zoom or self.point is None or self.point2 is None:
            return
        pen = QPen(QColor("white"))
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([5, 5])
        painter.setPen(pen)
        diff = self.point2 - self.point
        painter.drawRect(self.point.x(), self.point.y(), diff.x(), diff.y())
        pen = QPen(QColor("blue"))
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([5, 5])
        pen.setDashOffset(3)
        painter.setPen(pen)
        painter.drawRect(self.point.x(), self.point.y(), diff.x(), diff.y())


def create_tool_button(text, icon):
    res = QToolButton()
    # res.setIconSize(canvas_icon_size)
    if icon is None:
        res.setText(text)
    else:
        res.setToolTip(text)
        if isinstance(icon, str):
            res.setIcon(QIcon(os.path.join(icons_dir, icon)))
        else:
            res.setIcon(icon)
    return res


class ChanelColor(QWidget):
    def __init__(self, num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num = num
        self.check_box = QCheckBox(self)
        self.color_list = QComboBox(self)
        self.color_list.addItems(list(color_maps.keys()))
        num2 = num % len(default_colors)
        pos = list(color_maps.keys()).index(default_colors[num2])
        self.color_list.setCurrentIndex(pos)
        layout = QHBoxLayout()
        # noinspection PyArgumentList
        layout.addWidget(self.check_box)
        # noinspection PyArgumentList
        layout.addWidget(self.color_list)
        self.setLayout(layout)

    def channel_visible(self):
        return self.check_box.isChecked()

    def colormap_name(self):
        return str(self.color_list.currentText())

    def register(self, fun: Callable):
        # noinspection PyUnresolvedReferences
        self.color_list.currentIndexChanged.connect(fun)
        self.check_box.stateChanged.connect(fun)

    def setVisible(self, val):
        super().setVisible(val)
        self.check_box.setChecked(val)

    def set_list(self, colormap_list):
        """
        :type colormap_list: list[str]
        :param colormap_list:
        :return:
        """
        text = str(self.color_list.currentText())
        try:
            index = colormap_list.index(text)
        except ValueError:
            index = -1
        if index != -1:
            self.color_list.blockSignals(True)
        self.color_list.clear()
        self.color_list.addItems(list(text))
        if index != -1:
            self.color_list.setCurrentIndex(index)
            self.blockSignals(False)


class ImageView(QWidget):
    position_changed = Signal([int, int, int], [int, int])
    component_clicked = Signal(int)
    text_info_change = Signal(str)

    image_canvas = ImageCanvas  # can be used to customize canvas. eg. add more signals
    hide_signal = Signal(bool)

    # zoom_changed = Signal(float, float, float)

    def __init__(self, settings: BaseSettings, channel_property: ChannelProperty, name: str):
        # noinspection PyArgumentList
        super().__init__()
        self._settings: BaseSettings = settings
        self.channel_property = channel_property
        self.exclude_btn_list = []
        self.image_state = ImageShowState(settings, name)
        self.channel_control = ColorComboBoxGroup(settings, name, channel_property, height=30)
        self._channel_control_top = True
        self.image_area = MyScrollArea(self.image_state, self.image_canvas)
        self.reset_button = create_tool_button("Reset zoom", "zoom-original.png")
        self.reset_button.clicked.connect(self.reset_image_size)
        self.zoom_button = create_tool_button("Zoom", "zoom-select.png")
        self.zoom_button.toggled.connect(self.image_state.set_zoom)
        self.zoom_button.setCheckable(True)
        self.zoom_button.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.component = None
        crop = QAction("Crop", self.zoom_button)
        # crop.triggered.connect(self.crop_view)
        self.zoom_button.addAction(crop)
        self.move_button = create_tool_button("Move", "transform-move.png")
        self.move_button.toggled.connect(self.image_state.set_move)
        self.move_button.setCheckable(True)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.setSpacing(0)
        self.btn_layout.setContentsMargins(0, 0, 0, 0)
        # noinspection PyArgumentList
        self.btn_layout.addWidget(self.reset_button)
        # noinspection PyArgumentList
        self.btn_layout.addWidget(self.zoom_button)
        # noinspection PyArgumentList
        self.btn_layout.addWidget(self.move_button)
        # noinspection PyArgumentList
        self.btn_layout.addWidget(self.channel_control, 1)
        self.btn_layout2 = QHBoxLayout()

        self.stack_slider = QScrollBar(Qt.Horizontal)
        self.stack_slider.valueChanged.connect(self.paint_layer)
        self.stack_slider.valueChanged.connect(self.change_layer)
        self.time_slider = QScrollBar(Qt.Vertical)
        self.time_slider.valueChanged.connect(self.paint_layer)
        self.time_slider.valueChanged.connect(self.change_time)
        self.stack_layer_info = QLabel()
        self.time_layer_info = QLabel()
        self.time_layer_info.setAlignment(Qt.AlignCenter)
        self.tmp_image = None
        self.labels_layer = None
        self.image_shape = QSize(1, 1)

        main_layout = QGridLayout()
        main_layout.setSpacing(0)
        self.btn_layout.setSpacing(10)
        # main_layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(self.btn_layout, 0, 1)
        main_layout.addLayout(self.btn_layout2, 1, 1)
        time_slider_layout = QVBoxLayout()
        time_slider_layout.setContentsMargins(0, 0, 0, 0)
        # noinspection PyArgumentList
        time_slider_layout.addWidget(self.time_layer_info)
        # noinspection PyArgumentList
        time_slider_layout.addWidget(self.time_slider, 1)
        main_layout.addLayout(time_slider_layout, 2, 0)
        # noinspection PyArgumentList
        main_layout.addWidget(self.image_area, 2, 1)
        stack_slider_layout = QHBoxLayout()
        stack_slider_layout.setContentsMargins(0, 0, 0, 0)
        # noinspection PyArgumentList
        stack_slider_layout.addWidget(self.stack_slider, 1)
        # noinspection PyArgumentList
        stack_slider_layout.addWidget(self.stack_layer_info)
        main_layout.addLayout(stack_slider_layout, 3, 1)

        self.setLayout(main_layout)
        self.exclude_btn_list.extend([self.zoom_button, self.move_button])
        self.zoom_button.clicked.connect(self.exclude_btn_fun)
        self.move_button.clicked.connect(self.exclude_btn_fun)

        self.image_state.parameter_changed.connect(self.paint_layer)
        self.image_area.pixmap.position_signal.connect(self.position_info)
        self.image_area.pixmap.leave_signal.connect(self.clean_text)
        self.position_changed[int, int, int].connect(self.info_text_pos)
        self.position_changed[int, int].connect(self.info_text_pos)
        settings.segmentation_changed.connect(self.set_labels)
        settings.segmentation_clean.connect(self.set_labels)
        settings.image_changed.connect(self.set_image)
        settings.labels_changed.connect(self.paint_layer)
        self.channel_control.coloring_update.connect(self.paint_layer)

    def resizeEvent(self, event: QResizeEvent):
        if event.size().width() > 500 and not self._channel_control_top:
            w = self.btn_layout2.takeAt(0).widget()
            self.btn_layout.takeAt(3)
            # noinspection PyArgumentList
            self.btn_layout.insertWidget(3, w)
            self._channel_control_top = True
        elif event.size().width() <= 500 and self._channel_control_top:
            w = self.btn_layout.takeAt(3).widget()
            self.btn_layout.insertStretch(3, 1)
            # noinspection PyArgumentList
            self.btn_layout2.insertWidget(0, w)
            self._channel_control_top = False

    def update_channels_coloring(self, new_image: bool):
        if not new_image:
            self.paint_layer()

    def exclude_btn_fun(self):
        sender = self.sender()
        for el in self.exclude_btn_list:
            if el != sender:
                el.setChecked(False)

    def clean_text(self):
        self.text_info_change.emit("")

    def info_text_pos(self, *pos):
        if self.tmp_image is None:
            return
        try:
            brightness = self.tmp_image[pos if len(pos) == self.tmp_image.ndim - 1 else pos[1:]]
        except IndexError:
            return
        pos2 = list(pos)
        pos2[0] += 1
        if isinstance(brightness, collections.Iterable):
            res_brightness = []
            for i, b in enumerate(brightness):
                if self.channel_control.active_channel(i):
                    res_brightness.append(b)
            brightness = ", ".join(map(str, res_brightness))
        res_text = f"Position: {tuple(pos2)}, Brightness: {brightness}"
        if self.labels_layer is not None:
            comp = self.labels_layer[pos]
            self.component = comp
            if comp == 0:
                comp = "none"
                self.component = None
            else:
                try:
                    comp = "{} (size: {})".format(comp, self._settings.sizes[comp])
                except IndexError:
                    self.labels_layer = None
                    self.paint_layer()
                    return
            res_text += f", component {comp}"

        if self._settings.mask is not None:
            res_text += f", mask {self._settings.mask[pos]}"

        self.text_info_change.emit(res_text)

    def position_info(self, point, size):
        """
        :type point: QPoint
        :type size: QSize
        :param point:
        :return:
        """
        x = int(point.x() / size.width() * self.image_shape.width())
        y = int(point.y() / size.height() * self.image_shape.height())
        self.position_changed[int, int, int].emit(self.stack_slider.value(), y, x)

    def get_control_view(self):
        # type: () -> ImageShowState
        return self.image_state

    def reset_image_size(self):
        self.image_area.reset_image()

    def change_layer(self, num):
        self.stack_layer_info.setText("{} of {}".format(num + 1, self.image.layers))

    def change_time(self, num):
        self.time_layer_info.setText("{}\nof\n{}".format(num + 1, self.image.times))

    def get_layer(self):
        """ Function to overwrite if need create viewer in other dimensions"""
        return self.image.get_layer(self.time_slider.value(), self.stack_slider.value())

    def paint_layer(self):
        if self.image is None:
            return
        try:
            img = np.copy(self.get_layer())
        except IndexError:
            print(self.sender())
            raise
        color_list = self.channel_control.current_colormaps
        borders = self._settings.border_val[:]
        for i, p in enumerate(self.channel_control.get_limits()):
            if p is not None:
                borders[i] = p
        for i, (use, radius) in enumerate(self.channel_control.get_filter()):
            if use != NoiseFilterType.No and color_list[i] is not None and radius > 0:
                img[..., i] = apply_filter(use, img[..., i], radius)
        im = color_image_fun(img, color_list, borders)
        self.add_labels(im)
        self.add_mask(im)
        self.image_area.set_image(im, True)
        self.tmp_image = np.array(img)

    def add_mask(self, im):
        pass

    def add_labels(self, im):
        if self.labels_layer is not None and self.image_state.show_label != LabelEnum.Not_show:
            # TODO fix to support time
            layers = self.labels_layer[self.stack_slider.value()]
            components_mask = self._settings.components_mask()
            if self.image_state.show_label == LabelEnum.Show_results:
                components_mask[1:] = 1
            add_labels(
                im,
                layers,
                self.image_state.opacity,
                self.image_state.only_borders,
                int((self.image_state.borders_thick - 1) / 2),
                components_mask,
                self._settings.label_colors,
            )
        return im

    @property
    def image(self) -> Image:
        return self._settings.image

    def set_image(self):
        """
        function which set sliders and image size. If create viewers in other dimensions need to overwrite
        """
        self.labels_layer = None
        self.image_shape = QSize(self.image.plane_shape[1], self.image.plane_shape[0])
        self.stack_slider.blockSignals(True)
        self.stack_slider.setRange(0, self.image.layers - 1)
        self.stack_slider.setValue(self.image.layers // 2)
        self.stack_slider.blockSignals(False)
        self.time_slider.blockSignals(True)
        self.time_slider.setRange(0, self.image.times - 1)
        self.time_slider.setValue(self.image.times // 2)
        self.time_slider.blockSignals(False)
        self.change_layer(self.image.layers // 2)
        self.change_time(self.image.times // 2)
        self.channel_control.set_channels(self.image.channels)
        self.paint_layer()
        self.stack_slider.setHidden(self.image.layers == 1)
        self.stack_layer_info.setHidden(self.image.layers == 1)
        self.time_slider.setHidden(self.image.times == 1)
        self.time_layer_info.setHidden(self.image.times == 1)
        # self.image_area.set_image(image)

    @Slot()
    @Slot(np.ndarray)
    def set_labels(self, labels: Optional[np.ndarray] = None):
        if isinstance(labels, np.ndarray) and labels.size == 0:
            labels = None
        self.labels_layer = labels
        self.paint_layer()

    def hideEvent(self, a0: QtGui.QHideEvent) -> None:
        self.hide_signal.emit(True)

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        self.hide_signal.emit(False)


class MyScrollArea(QScrollArea):
    """
    :type image_ratio: float
    :param image_ratio: image width/height ratio
    :type zoom_scale: float
    :param zoom_scale: zoom scale
    """

    # resize_area = Signal(QSize)

    zoom_changed = Signal()

    def __init__(self, local_settings, image_canvas: Type[ImageCanvas], *args, **kwargs):
        """
        :type local_settings: ImageShowState
        :param local_settings:
        :param args:
        :param kwargs:
        """
        super(MyScrollArea, self).__init__(*args, **kwargs)
        self.local_settings = local_settings
        self.setAlignment(Qt.AlignCenter)
        self.clicked = False
        self.prev_pos = None
        self.pixmap: ImageCanvas = image_canvas(local_settings)
        self.pixmap.setScaledContents(True)
        self.pixmap.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.pixmap.setBackgroundRole(QPalette.Base)
        self.pixmap.setScaledContents(True)
        self.pixmap.zoom_mark.connect(self.zoom_image)
        self.setBackgroundRole(QPalette.Dark)
        self.setWidget(self.pixmap)
        # self.image_ratio = 1
        self.zoom_scale = 1
        self.max_zoom = 20
        self.image_size = QSize(1, 1)
        self.horizontal_ratio = False, 1
        self.vertical_ratio = False, 1
        self.y_mid = None
        self.x_mid = None
        # self.setWidgetResizable(True)
        self.horizontalScrollBar().rangeChanged.connect(self.horizontal_range_changed)
        self.verticalScrollBar().rangeChanged.connect(self.vertical_range_changed)
        # self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.timer_id = 0

    def horizontal_range_changed(self, min_val, max_val):
        if self.x_mid is not None and self.sender().maximum() > 0:
            diff = self.widget().size().width() - (max_val - min_val)
            self.sender().setValue(self.x_mid - diff / 2)
            self.x_mid = None

    def vertical_range_changed(self, min_val, max_val):
        if self.y_mid is not None and self.sender().maximum() > 0:
            diff = self.widget().size().height() - (max_val - min_val)
            self.sender().setValue(self.y_mid - diff / 2)
            self.y_mid = None

    def get_ratio_factor(self, size=None):
        if size is None:
            size = self.size()
        pixmap_ratio = self.pixmap.image_size.width() / self.pixmap.image_size.height()
        area_ratio = size.width() / size.height()
        if pixmap_ratio < area_ratio:
            ratio = size.height() / self.pixmap.image_size.height()
        else:
            ratio = size.width() / self.pixmap.image_size.width()
        ratio = ratio * self.zoom_scale
        # noinspection PyTypeChecker
        return ratio

    def zoom_image(self, point1, point2):
        """
        :type point1: QPoint
        :type point2: QPoint
        :param point1:
        :param point2:
        :return:
        """
        x_width = abs(point1.x() - point2.x())
        y_width = abs(point1.y() - point2.y())
        if x_width < 10 or y_width < 10:
            return
        x_ratio = self.width() / x_width
        y_ratio = self.height() / y_width
        scale_ratio = min(x_ratio, y_ratio)

        if self.zoom_scale * scale_ratio > self.max_zoom:
            scale_ratio = self.max_zoom / self.zoom_scale
            self.zoom_scale = self.max_zoom
        else:
            self.zoom_scale *= scale_ratio
        # print(f"Zoom scale: {self.zoom_scale}")
        if scale_ratio == 1:
            return
        ratio = self.get_ratio_factor()
        # noinspection PyTypeChecker
        final_size = QSize(self.pixmap.image_size * ratio - QSize(2, 2))
        self.y_mid = (point1.y() + point2.y()) / 2 * scale_ratio
        self.x_mid = (point1.x() + point2.x()) / 2 * scale_ratio
        self.pixmap.resize(final_size)
        self.zoom_changed.emit()

    @property
    def image_ratio(self):
        return self.widget().image_ratio

    def reset_image(self):
        x = self.size().width() - 2
        y = self.size().height() - 2
        if float(x) > y * self.image_ratio:
            x = int(y * self.image_ratio)
        else:
            y = int(x / self.image_ratio)
        self.pixmap.resize(x, y)
        self.zoom_scale = 1
        self.x_mid = None
        self.y_mid = None
        self.zoom_changed.emit()

    def set_image(self, im, keep_size=False):
        self.widget().set_image(im)
        if not keep_size:
            self.reset_image()
            # self.widget().adjustSize()

    def mousePressEvent(self, event):
        self.clicked = True
        self.prev_pos = event.x(), event.y()

    def mouseReleaseEvent(self, event):
        self.clicked = False
        self.prev_pos = None

    def mouseMoveEvent(self, event):
        if not self.local_settings.move or self.prev_pos is None:
            return
        x, y = event.x(), event.y()
        x_dif, y_dif = self.prev_pos[0] - x, self.prev_pos[1] - y
        h_bar = self.horizontalScrollBar()
        h_bar.setValue(h_bar.value() + x_dif)
        v_bar = self.verticalScrollBar()
        v_bar.setValue(v_bar.value() + y_dif)
        self.prev_pos = x, y

    def showEvent(self, event):
        super(MyScrollArea, self).showEvent(event)
        if not event.spontaneous():
            scroll_bar = self.horizontalScrollBar()
            n_val = (scroll_bar.minimum() + scroll_bar.maximum()) // 2
            scroll_bar.setValue(n_val)
            scroll_bar = self.verticalScrollBar()
            n_val = (scroll_bar.minimum() + scroll_bar.maximum()) // 2
            scroll_bar.setValue(n_val)

    @staticmethod
    def calculate_shift(pixmap_len, self_len, pix_ratio, cursor_ratio, scroll_bar: QScrollBar):
        if pixmap_len - self_len > 0:
            scroll_bar.setValue(pixmap_len * pix_ratio - self_len * cursor_ratio)

    def resize_pixmap(self):
        ratio = self.get_ratio_factor()
        # noinspection PyTypeChecker
        final_size = QSize(self.pixmap.image_size * ratio - QSize(2, 2))

        if final_size == self.pixmap.size():
            return
        else:
            self.pixmap.resize(final_size)

    def center_pixmap(self):
        x_cord = (self.width() - 2 - self.pixmap.width()) // 2
        y_cord = (self.height() - 2 - self.pixmap.height()) // 2
        self.pixmap.move(x_cord, y_cord)

    def resizeEvent(self, event):
        # super(MyScrollArea, self).resizeEvent(event)
        self.pixmap.point = None
        if self.x_mid is None:
            self.x_mid = -self.widget().pos().x() + (self.get_width(event.oldSize().width())) / 2
        if self.y_mid is None:
            self.y_mid = -self.widget().pos().y() + (self.get_height(event.oldSize().height())) / 2
        old_ratio = self.get_ratio_factor(event.oldSize())
        new_ratio = self.get_ratio_factor(event.size())
        scalar = new_ratio / old_ratio
        self.x_mid *= scalar
        self.y_mid *= scalar
        if self.size().width() - 2 > self.pixmap.width() and self.size().height() - 2 > self.pixmap.height():
            self.reset_image()
        elif not (self.size().width() - 2 < self.pixmap.width() or self.size().height() - 2 < self.pixmap.height()):
            self.center_pixmap()
        else:
            self.resize_pixmap()

    def get_width(self, width=None):
        if width is None:
            width = self.width()
        if self.verticalScrollBar().isVisible():
            return width - self.verticalScrollBar().size().width()
        else:
            return width

    def get_height(self, height=None):
        if height is None:
            height = self.height()
        if self.horizontalScrollBar().isVisible():
            return height - self.horizontalScrollBar().size().height()
        else:
            return height

    def wheelEvent(self, event: QWheelEvent):
        # noinspection PyTypeChecker
        # if not (QApplication.keyboardModifiers() & Qt.ControlModifier) == Qt.ControlModifier:
        #     return
        delta = event.angleDelta().y()
        if abs(delta) > max_step:
            delta = max_step * (delta / abs(delta))
        scale_mod = step ** delta
        if scale_mod == 1 or (scale_mod > 1 and self.zoom_scale == self.max_zoom):
            return
        if self.zoom_scale * scale_mod > self.max_zoom:
            self.zoom_scale = self.max_zoom
        elif self.zoom_scale * scale_mod < 1:
            return
        else:
            self.zoom_scale *= scale_mod

        x_modify = self.pixmap.pos().x() if self.pixmap.pos().x() > 0 else 0
        y_modify = self.pixmap.pos().y() if self.pixmap.pos().y() > 0 else 0
        x_pos = event.x() - self.widget().pos().x() + x_modify
        y_pos = event.y() - self.widget().pos().y() + y_modify
        x_ratio = x_pos / self.widget().size().width()
        y_ratio = y_pos / self.widget().size().height()
        ratio = self.get_ratio_factor()
        # noinspection PyTypeChecker
        final_size = QSize(self.pixmap.image_size * ratio - QSize(2, 2))
        x_pos_new = final_size.width() * x_ratio
        y_pos_new = final_size.height() * y_ratio
        self.x_mid = x_pos_new - event.x() + (self.get_width()) / 2
        self.y_mid = y_pos_new - event.y() + (self.get_height()) / 2
        print(self.pixmap.pos(), self.x_mid, self.y_mid)

        if self.size().width() - 2 > self.pixmap.width() and self.size().height() - 2 > self.pixmap.height():
            # print("B")
            self.reset_image()
        else:
            # print("C", self.pixmap.size())
            self.resize_pixmap()


class ColorBar(QLabel):
    def __init__(self, settings: ViewSettings, image_view: Union[List[ImageView], ImageView]):
        super().__init__()
        self.image_view = image_view
        self._settings = settings
        self.image = None
        if isinstance(image_view, list):
            for el in image_view:
                el.channel_control.change_channel.connect(self.update_colormap)
        else:
            image_view.channel_control.change_channel.connect(self.update_colormap)
        self.range = None
        self.round_range = None
        self.setFixedWidth(80)

    def update_colormap(self, name, channel_id):
        fixed_range = self._settings.get_from_profile(f"{name}.lock_{channel_id}", False)
        if fixed_range:
            self.range = self._settings.get_from_profile(f"{name}.range_{channel_id}")
        else:
            self.range = self._settings.border_val[channel_id]
        cmap = self._settings.colormap_dict[self._settings.get_channel_info(name, channel_id)][0]

        round_factor = self.round_base(self.range[1])
        self.round_range = (
            int(round(self.range[0] / round_factor) * round_factor),
            int(round(self.range[1] / round_factor) * round_factor),
        )
        if self.round_range[0] < self.range[0]:
            self.round_range = self.round_range[0] + round_factor, self.round_range[1]
        if self.round_range[1] > self.range[1]:
            self.round_range = self.round_range[0], self.round_range[1] - round_factor
        # print(self.range, self.round_range)
        img = color_image_fun(np.linspace(0, 256, 512).reshape((1, 512, 1))[:, ::-1], [cmap], [(0, 256)])
        self.image = NumpyQImage(np.swapaxes(img, 0, 1))
        self.repaint()

    @staticmethod
    def round_base(val):
        if val > 10000:
            return 1000
        if val > 1000:
            return 100
        if val > 100:
            return 10
        return 1

    @staticmethod
    def number_of_marks(val):
        if val < 500:
            return 6
        if val > 1300:
            return 21
        return 11

    def paintEvent(self, event: QtGui.QPaintEvent):
        bar_width = 30

        if self.image is None:
            return

        rect = self.rect()
        number_of_marks = self.number_of_marks(rect.height())
        image_rect = QRect(rect.topLeft(), QSize(bar_width, rect.size().height()))
        painter = QPainter(self)
        old_font = painter.font()
        new_font = painter.font()
        new_font.setPointSizeF(new_font.pointSizeF() / 1.1)
        painter.setFont(new_font)
        painter.drawImage(image_rect, self.image)
        if self.range[1] == self.range[0]:
            painter.drawText(bar_width + 5, 20, f"{self.range[1]}")
            painter.drawText(bar_width + 5, rect.size().height(), f"{self.range[1]}")
            painter.setFont(old_font)
            return
        start_prop = 1 - (self.round_range[0] - self.range[0]) / (self.range[1] - self.range[0])
        end_prop = 1 - (self.round_range[1] - self.range[0]) / (self.range[1] - self.range[0])
        for pos, val in zip(
            np.linspace(10 + end_prop * rect.size().height(), start_prop * rect.size().height(), number_of_marks),
            np.linspace(self.round_range[1], self.round_range[0], number_of_marks, dtype=np.uint32),
        ):
            painter.drawText(bar_width + 5, pos, f"{val}")
        painter.setFont(old_font)
        # print(self.image.shape)


class ImageViewWithMask(ImageView):
    def __init__(self, settings: BaseSettings, channel_property: ChannelProperty, name: str):
        super().__init__(settings, channel_property, name)
        self.mask_show = QCheckBox()
        self.mask_label = QLabel("Mask:")
        self.btn_layout.addWidget(self.mask_label)
        self.btn_layout.addWidget(self.mask_show)
        self.mask_prop = (
            np.array(self._settings.get_from_profile("mask_presentation_color", [255, 255, 255])),
            self._settings.get_from_profile("mask_presentation_opacity", 1),
        )
        self.mask_show.setDisabled(True)
        self.mask_label.setDisabled(True)
        settings.mask_changed.connect(self.mask_changed)
        self.mask_show.stateChanged.connect(self.paint_layer)

    def event(self, event: QtCore.QEvent):
        if event.type() == QEvent.WindowActivate:
            if self.mask_show.isChecked():
                color = np.array(self._settings.get_from_profile("mask_presentation_color", [255, 255, 255]))
                opacity = self._settings.get_from_profile("mask_presentation_opacity", 1)
                if np.any(color != self.mask_prop[0]) or opacity != self.mask_prop[1]:
                    self.mask_prop = color, opacity
                    self.paint_layer()
        return super().event(event)

    def mask_changed(self):
        self.mask_show.setDisabled(self._settings.mask is None)
        self.mask_label.setDisabled(self._settings.mask is None)
        if self._settings.mask is None:
            self.mask_show.setChecked(False)
        elif self.mask_show.isChecked():
            self.paint_layer()

    def add_mask(self, im):
        if not self.mask_show.isChecked() or self._settings.mask is None:
            return
        mask_layer = self._settings.mask[self.stack_slider.value()]
        mask_layer = mask_layer.astype(np.bool)

        if self.mask_prop[1] == 1:
            im[~mask_layer] = self.mask_prop[0]
        else:
            im[~mask_layer] = (1 - self.mask_prop[1]) * im[~mask_layer] + self.mask_prop[1] * self.mask_prop[0]

    def set_image(self):
        super().set_image()
        self.mask_changed()
