from __future__ import division, print_function

import os
from math import log
from typing import List, Union

import numpy as np
from qtpy import QtGui
from qtpy.QtCore import QRect, QSize
from qtpy.QtGui import QIcon, QPainter
from qtpy.QtWidgets import QLabel, QToolButton

from PartSeg.common_gui.numpy_qimage import NumpyQImage
from PartSegCore.class_generator import enum_register
from PartSegCore.color_image import color_image_fun
from PartSegData import icons_dir

from ..common_backend.base_settings import ViewSettings
from .napari_image_view import ImageView, LabelEnum

canvas_icon_size = QSize(20, 20)
step = 1.01
max_step = log(1.2, step)

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


def create_tool_button(text: str, icon: Union[str, QIcon]) -> QToolButton:
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
        gamma = self._settings.get_from_profile(f"{name}.gamma_value_{channel_id}", 1)
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
        data = np.linspace(0, 1, 512)
        data = (data ** gamma) * 255
        img = color_image_fun(data.reshape((1, 512, 1))[:, ::-1], [cmap], [(0, 256)])
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
