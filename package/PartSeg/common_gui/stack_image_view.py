from math import log
from typing import List, Union

import numpy as np
from napari.utils import Colormap
from napari.utils.colormaps import make_colorbar
from qtpy import QtGui
from qtpy.QtCore import QRect, QSize
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import QLabel

from PartSeg.common_backend.base_settings import ViewSettings
from PartSeg.common_gui.napari_image_view import ImageView
from PartSeg.common_gui.numpy_qimage import NumpyQImage

canvas_icon_size = QSize(20, 20)
step = 1.01
max_step = log(1.2, step)


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
        cmap = self._settings.colormap_dict[self._settings.get_channel_colormap_name(name, channel_id)][0]

        round_factor = self.round_base(self.range[1])
        self.round_range = (
            int(round(self.range[0] / round_factor) * round_factor),
            int(round(self.range[1] / round_factor) * round_factor),
        )
        if self.round_range[0] < self.range[0]:
            self.round_range = self.round_range[0] + round_factor, self.round_range[1]
        if self.round_range[1] > self.range[1]:
            self.round_range = self.round_range[0], self.round_range[1] - round_factor
        data = np.linspace(0, 1, 512)
        interpolated = cmap.map(data)
        data = data**gamma
        colormap = Colormap(interpolated, controls=data)
        self.image = NumpyQImage(np.array(make_colorbar(colormap, size=(512, 1), horizontal=False)[::-1]))
        self.repaint()

    @staticmethod
    def round_base(val):
        if val > 10000:
            return 1000
        if val > 1000:
            return 100
        return 10 if val > 100 else 1

    @staticmethod
    def number_of_marks(val):
        if val < 500:
            return 6
        return 21 if val > 1300 else 11

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
            painter.drawText(bar_width + 5, int(pos), f"{val}")
        painter.setFont(old_font)
