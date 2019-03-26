from __future__ import division, print_function

import collections
import os
from math import log
from typing import Type

import numpy as np
from qtpy import QtGui
from qtpy.QtCore import QRect, QTimerEvent, QSize, QObject, Signal, QPoint, Qt, QEvent, Slot
from qtpy.QtGui import QShowEvent, QWheelEvent, QPainter, QPen, QColor, QPalette, QPixmap, QImage, QIcon
from qtpy.QtWidgets import QScrollBar, QLabel, QGridLayout
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, \
    QScrollArea, QSizePolicy, QToolButton, QAction, QApplication, \
    QSlider, QCheckBox, QComboBox
from scipy.ndimage import gaussian_filter

from ..utils.color_image import color_image, add_labels
from ..utils.color_image.color_image_base import color_maps
from ..utils.colors import default_colors
from ..utils.global_settings import static_file_folder
from ..project_utils_qt.settings import ViewSettings, BaseSettings
from PartSeg.tiff_image import Image
from .channel_control import ChannelControl

canvas_icon_size = QSize(20, 20)
step = 1.01
max_step = log(1.2, step)


class ImageState(QObject):
    parameter_changed = Signal()
    opacity_changed = Signal(float)
    show_label_changed = Signal(bool)

    def __init__(self, settings: ViewSettings):
        super(ImageState, self).__init__()
        self.settings = settings
        self.zoom = False
        self.move = False
        self.opacity = settings.get_from_profile("image_state.opacity", 1)
        self.show_label = settings.get_from_profile("image_state.show_label", 1)
        # 0 - no show, 1 - show all, 2 - show chosen
        self.only_borders = settings.get_from_profile("image_state.only_border", True)
        self.borders_thick = settings.get_from_profile("image_state.border_thick", 1)

    def set_zoom(self, val):
        self.zoom = val

    def set_move(self, val):
        self.move = val

    def set_borders(self, val):
        if self.only_borders != val:
            self.settings.set_in_profile("image_state.only_border", val)
            self.only_borders = val
            self.parameter_changed.emit()

    def set_borders_thick(self, val):
        if val != self.borders_thick:
            self.settings.set_in_profile("image_state.border_thick", val)
            self.borders_thick = val
            self.parameter_changed.emit()

    def set_opacity(self, val):
        if self.opacity != val:
            self.settings.set_in_profile("image_state.opacity", val)
            self.opacity = val
            self.parameter_changed.emit()

    def components_change(self):
        if self.show_label == 2:
            self.parameter_changed.emit()

    def set_show_label(self, val):
        if self.show_label != val:
            self.settings.set_in_profile("image_state.show_label", val)
            self.show_label = val
            self.parameter_changed.emit()


class ImageCanvas(QLabel):
    zoom_mark = Signal(QPoint, QPoint)
    position_signal = Signal(QPoint, QSize)
    click_signal = Signal(QPoint, QSize)
    leave_signal = Signal()

    def __init__(self, local_settings):
        """
        :type local_settings: ImageState
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

    def set_image(self, im, paint):
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
        self.my_pixmap = QPixmap.fromImage(im2) # .scaled(self.width(), self.height(), Qt.KeepAspectRatio))
        self.repaint()

    def leaveEvent(self, a0: QEvent):
        self.point = None
        self.point2 = None
        self.leave_signal.emit()

    def mousePressEvent(self, event):
        """
        :type event: QMouseEvent
        :param event:
        :return:
        """
        super().mousePressEvent(event)
        if self.local_settings.zoom:
            self.point = event.pos()
        elif not self.local_settings.move:
            self.click_signal.emit(event.pos(), self.size())

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.local_settings.zoom and self.point is not None:
            self.point2 = event.pos()
            self.update()
        self.position_signal.emit(event.pos(), self.size())

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.local_settings.zoom and self.point is not None and self.point2 is not None:
            diff = self.point2 - self.point
            if abs(diff.x()) and abs(diff.y()):
                self.zoom_mark.emit(self.point, self.point2)
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

    def resizeEvent(self, event: QtGui.QResizeEvent):
        pass
        # print("[resize]", self.size(), self.parent().size(), event.oldSize(), event.size())
        # self.paint_image()


def get_scroll_bar_proportion(scroll_bar):
    """
    :type scroll_bar: QScrollBar
    :param scroll_bar:
    :return: float
    """
    dist = (scroll_bar.maximum() - scroll_bar.minimum())
    if dist == 0:
        return 0.5
    else:
        return float(scroll_bar.value()) / dist


def set_scroll_bar_proportion(scroll_bar, proportion):
    """
    :type scroll_bar: QScrollBar
    :type proportion: float
    :param scroll_bar:
    """
    scroll_bar.setValue(int((scroll_bar.maximum() - scroll_bar.minimum()) * proportion))


def create_tool_button(text, icon):
    res = QToolButton()
    # res.setIconSize(canvas_icon_size)
    if icon is None:
        res.setText(text)
    else:
        res.setToolTip(text)
        if isinstance(icon, str):
            res.setIcon(QIcon(os.path.join(static_file_folder, "icons", icon)))
        else:
            res.setIcon(icon)
    return res


class ChanelColor(QWidget):
    def __init__(self, num, *args, **kwargs):
        super(ChanelColor, self).__init__(*args, **kwargs)
        self.num = num
        self.check_box = QCheckBox(self)
        self.color_list = QComboBox(self)
        self.color_list.addItems(color_maps.keys())
        num2 = num % len(default_colors)
        pos = list(color_maps.keys()).index(default_colors[num2])
        self.color_list.setCurrentIndex(pos)
        layout = QHBoxLayout()
        layout.addWidget(self.check_box)
        layout.addWidget(self.color_list)
        self.setLayout(layout)

    def channel_visible(self):
        return self.check_box.isChecked()

    def colormap_name(self):
        return str(self.color_list.currentText())

    """def colormap(self, vmin, vmax):
        cmap = get_cmap(str(self.color_list.currentText()))
        norm = PowerNorm(1, vmin=vmin, vmax=vmax)
        return lambda x: cmap(norm(x))"""

    def register(self, fun):
        # noinspection PyUnresolvedReferences
        self.color_list.currentIndexChanged.connect(fun)
        self.check_box.stateChanged.connect(fun)

    def setVisible(self, val):
        super(ChanelColor, self).setVisible(val)
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
        self.color_list.addItems(text)
        if index != -1:
            self.color_list.setCurrentIndex(index)
            self.blockSignals(False)


class ImageView(QWidget):
    position_changed = Signal([int, int, int], [int, int])
    component_clicked = Signal(int)
    text_info_change = Signal(str)

    image_canvas = ImageCanvas

    # zoom_changed = Signal(float, float, float)

    def __init__(self, settings, channel_control: ChannelControl):
        """:type settings: ViewSettings"""
        super(ImageView, self).__init__()
        self._settings: BaseSettings = settings
        self.channel_control = channel_control
        self.exclude_btn_list = []
        self.image_state = ImageState(settings)
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
        self.btn_layout.addWidget(self.reset_button)
        self.btn_layout.addWidget(self.zoom_button)
        self.btn_layout.addWidget(self.move_button)
        self.btn_layout.addStretch(1)

        self.stack_slider = QSlider(Qt.Horizontal)
        self.stack_slider.valueChanged.connect(self.change_image)
        self.stack_slider.valueChanged.connect(self.change_layer)
        self.time_slider = QSlider(Qt.Vertical)
        self.time_slider.valueChanged.connect(self.change_image)
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
        time_slider_layout = QVBoxLayout()
        time_slider_layout.setContentsMargins(0, 0, 0, 0)
        time_slider_layout.addWidget(self.time_layer_info)
        time_slider_layout.addWidget(self.time_slider)
        main_layout.addLayout(time_slider_layout, 1, 0)
        main_layout.addWidget(self.image_area, 1, 1)
        stack_slider_layout = QHBoxLayout()
        stack_slider_layout.setContentsMargins(0, 0, 0, 0)
        stack_slider_layout.addWidget(self.stack_slider)
        stack_slider_layout.addWidget(self.stack_layer_info)
        main_layout.addLayout(stack_slider_layout, 2, 1)

        self.setLayout(main_layout)
        self.exclude_btn_list.extend([self.zoom_button, self.move_button])
        self.zoom_button.clicked.connect(self.exclude_btn_fun)
        self.move_button.clicked.connect(self.exclude_btn_fun)

        self.image_state.parameter_changed.connect(self.change_image)
        self.image_area.pixmap.position_signal.connect(self.position_info)
        self.image_area.pixmap.leave_signal.connect(self.clean_text)
        self.position_changed[int, int, int].connect(self.info_text_pos)
        self.position_changed[int, int].connect(self.info_text_pos)
        self.channel_control.coloring_update.connect(self.update_channels_coloring)

        settings.segmentation_changed.connect(self.set_labels)
        settings.segmentation_clean.connect(self.set_labels)

    def update_channels_coloring(self, new_image: bool):
        if not new_image:
            self.change_image()

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
            brightness = res_brightness
            if len(brightness) == 1:
                brightness = brightness[0]
        if self.labels_layer is not None:
            comp = self.labels_layer[pos]
            self.component = comp
            if comp == 0:
                comp = "none"
                self.component = None
            else:
                comp = "{} (size: {})".format(comp, self._settings.sizes[comp])
            self.text_info_change.emit("Position: {}, Brightness: {}, component {}".format(
                tuple(pos2), brightness, comp))
        else:
            self.text_info_change.emit("Position: {}, Brightness: {}".format(tuple(pos2), brightness))

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
        # type: () -> ImageState
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

    def change_image(self):
        if self.image is None:
            return
        img = np.copy(self.get_layer())
        color_maps = self.channel_control.current_colors
        borders = self._settings.border_val[:]
        for i, p in enumerate(self.channel_control.get_limits()):
            if p is not None:
                borders[i] = p
        for i, (use, radius) in enumerate(self.channel_control.get_gauss()):
            if use and color_maps[i] is not None and radius > 0:
                img[..., i] = gaussian_filter(img[..., i], radius)
        im = color_image(img, color_maps, borders)
        self.add_labels(im)
        self.add_mask(im)
        self.image_area.set_image(im, True)
        self.tmp_image = np.array(img)

    def add_mask(self, im):
        pass

    def add_labels(self, im):
        if self.labels_layer is not None and self.image_state.show_label:
            # TODO fix to support time
            layers = self.labels_layer[self.stack_slider.value()]
            components_mask = self._settings.components_mask()
            if self.image_state.show_label == 1:
                components_mask[1:] = 1
            add_labels(im, layers, self.image_state.opacity, self.image_state.only_borders,
                       int((self.image_state.borders_thick - 1) / 2), components_mask)
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
        self.change_image()
        self.change_layer(self.image.layers // 2)
        self.change_time(self.image.times // 2)
        self.stack_slider.setHidden(self.image.layers == 1)
        self.stack_layer_info.setHidden(self.image.layers == 1)
        self.time_slider.setHidden(self.image.times == 1)
        self.time_layer_info.setHidden(self.image.times == 1)
        # self.image_area.set_image(image)

    @Slot()
    @Slot(np.ndarray)
    def set_labels(self, labels=None):
        self.labels_layer = labels
        self.change_image()


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
        :type local_settings: ImageState
        :param local_settings:
        :param args:
        :param kwargs:
        """
        super(MyScrollArea, self).__init__(*args, **kwargs)
        self.local_settings = local_settings
        self.setAlignment(Qt.AlignCenter)
        self.clicked = False
        self.prev_pos = None
        self.pixmap = image_canvas(local_settings)
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
        if self.x_mid is not None and self.sender().isVisible():
            diff = self.widget().size().width() - (max_val - min_val)
            self.sender().setValue(self.x_mid - diff / 2)
            self.x_mid = None

    def vertical_range_changed(self, min_val, max_val):
        if self.y_mid is not None and self.sender().isVisible():
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
        self.widget().set_image(im, keep_size)
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
        if not self.local_settings.move:
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

    def calculate_shift(self, pixmap_len, self_len, pix_ratio, cursor_ratio, scroll_bar: QScrollBar):
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
            self.x_mid = - self.widget().pos().x() + (self.get_width(event.oldSize().width())) / 2
        if self.y_mid is None:
            self.y_mid = - self.widget().pos().y() + (self.get_height(event.oldSize().height())) / 2
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

    def timerEvent(self, a0: 'QTimerEvent'):
        # Some try to reduce number of repaint event
        self.killTimer(self.timer_id)
        self.timer_id = 0
        if self.size().width() - 2 > self.pixmap.width() and self.size().height() - 2 > self.pixmap.height():
            # print("B")
            self.reset_image()
        else:
            # print("C", self.pixmap.size())
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
        if not (QApplication.keyboardModifiers() & Qt.ControlModifier) == Qt.ControlModifier:
            return
        delta = event.angleDelta().y()
        if abs(delta) > max_step:
            delta = max_step * (delta / abs(delta))
        scale_mod = (step ** delta)
        if scale_mod == 1 or (scale_mod > 1 and self.zoom_scale == self.max_zoom):
            return
        if self.zoom_scale * scale_mod > self.max_zoom:
            self.zoom_scale = self.max_zoom
        elif self.zoom_scale * scale_mod < 1:
            return
        else:
            self.zoom_scale *= scale_mod

        x_pos = event.x() - self.widget().pos().x()
        y_pos = event.y() - self.widget().pos().y()
        x_ratio = x_pos / self.widget().size().width()
        y_ratio = y_pos / self.widget().size().height()
        ratio = self.get_ratio_factor()
        # noinspection PyTypeChecker
        final_size = QSize(self.pixmap.image_size * ratio - QSize(2, 2))
        x_pos_new = final_size.width() * x_ratio
        y_pos_new = final_size.height() * y_ratio
        self.x_mid = x_pos_new - event.x() + (self.get_width()) / 2
        self.y_mid = y_pos_new - event.y() + (self.get_height()) / 2

        if self.timer_id:
            self.killTimer(self.timer_id)
            self.timer_id = 0
        self.timer_id = self.startTimer(50)
        self.zoom_changed.emit()
        event.accept()


class ColorBar(QLabel):
    def __init__(self, settings: ViewSettings, channel_control: ChannelControl):
        super().__init__()
        self.channel_control = channel_control
        self._settings = settings
        self.image = None
        self.channel_control.channel_change.connect(self.update_colormap)
        self.range = None
        self.round_range = None
        self.setFixedWidth(80)
        # layout = QHBoxLayout()
        # layout.addWidget(QLabel("aaa"))
        # self.setLayout(layout)

    def update_colormap(self, channel_id):
        fixed_range = self._settings.get_from_profile(f"{self.channel_control.name}.lock_{channel_id}", False)
        if fixed_range:
            self.range = self._settings.get_from_profile(f"{self.channel_control.name}.range_{channel_id}")
        else:
            self.range = self._settings.border_val[channel_id]
        cmap = self._settings.get_from_profile(f"{self.channel_control.name}.cmap{channel_id}")
        round_factor = self.round_base(self.range[1])
        self.round_range = (int(round(self.range[0] / round_factor) * round_factor),
                            int(round(self.range[1] / round_factor) * round_factor))
        if self.round_range[0] < self.range[0]:
            self.round_range = self.round_range[0] + round_factor, self.round_range[1]
        if self.round_range[1] > self.range[1]:
            self.round_range = self.round_range[0], self.round_range[1] - round_factor
        # print(self.range, self.round_range)

        img = color_image(np.linspace(0, 256, 512).reshape((1, 512, 1))[:, ::-1], [cmap], [(0, 256)])
        self.image = QImage(img.data, 1, 512, img.dtype.itemsize * 3, QImage.Format_RGB888)
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

        rect = event.rect()
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
        for pos, val in zip(np.linspace(10 + end_prop * rect.size().height(), start_prop * rect.size().height(),
                                        number_of_marks),
                            np.linspace(self.round_range[1], self.round_range[0], number_of_marks, dtype=np.uint32)):
            painter.drawText(bar_width + 5, pos, f"{val}")
        painter.setFont(old_font)
        # print(self.image.shape)
